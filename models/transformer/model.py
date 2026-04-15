import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from models.encoder.model import TripleEncoder


class CrossModalTransformer(nn.Module):
    # cross-modal transformer -- fuses 3 sensor embeddings using attention
    # input:  (batch, 3, embedding_dim) -- one embedding per sensor
    # output: (batch, fused_dim)        -- single fused representation

    def __init__(self, embedding_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()

        # positional encoding -- tells the transformer which sensor is which
        # sensor 0 = A (Micro80), sensor 1 = B (F50A), sensor 2 = C (Micro200HF)
        self.pos_encoding = nn.Parameter(torch.randn(1, 3, embedding_dim) * 0.02)

        # transformer encoder layers -- self attention across 3 sensor tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model     = embedding_dim,
            nhead       = num_heads,
            dim_feedforward = embedding_dim * 4,  # standard 4x feedforward expansion
            dropout     = dropout,
            batch_first = True,   # expects (batch, seq, dim) not (seq, batch, dim)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # layer norm after transformer
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # x shape: (batch, 3, embedding_dim) -- 3 sensor embeddings
        # add positional encoding so transformer knows which sensor is which
        x = x + self.pos_encoding

        # transformer self-attention across 3 sensor tokens
        x = self.transformer(x)   # (batch, 3, embedding_dim)
        x = self.norm(x)

        # aggregate 3 sensor outputs into one fused representation
        # mean pooling across the 3 sensors
        fused = x.mean(dim=1)     # (batch, embedding_dim)
        return fused


class JointFaultTransformer(nn.Module):
    # full model -- encoder + transformer + classification head
    # Phase 2 supervised fine-tuning model
    # input:  (batch, 3, 10000) -- raw windowed signal
    # output: (batch, num_classes) -- 7 loosening level logits

    def __init__(
        self,
        embedding_dim   = 128,
        num_heads       = 4,
        num_layers      = 2,
        num_classes     = 7,
        dropout         = 0.1,
        encoder_weights = None,   # path to SimCLR pre-trained encoder weights
    ):
        super().__init__()

        # encoder -- pre-trained in Phase 1
        self.encoder = TripleEncoder(embedding_dim=embedding_dim)

        # load pre-trained encoder weights if provided
        if encoder_weights is not None:
            state = torch.load(encoder_weights, map_location='cpu')
            self.encoder.load_state_dict(state)
            print(f'Loaded pre-trained encoder from {encoder_weights}')

        # cross-modal transformer -- fuses 3 sensor embeddings
        self.transformer = CrossModalTransformer(
            embedding_dim   = embedding_dim,
            num_heads       = num_heads,
            num_layers      = num_layers,
            dropout         = dropout,
        )

        # classification head -- maps fused representation to 7 class logits
        # no softmax here -- CrossEntropyLoss applies it internally
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, 3, 10000)

        # step 1 -- encode each sensor independently
        embeddings = self.encoder(x)       # (batch, 3, embedding_dim)

        # step 2 -- fuse across sensors with cross-modal transformer
        fused = self.transformer(embeddings)  # (batch, embedding_dim)

        # step 3 -- classify into 7 loosening levels
        logits = self.classifier(fused)    # (batch, num_classes)
        return logits

    def predict(self, x):
        # convenience method for inference -- returns class index and probabilities
        # x shape: (batch, 3, 10000)
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)                          # (batch, 7)
            probs  = torch.softmax(logits, dim=1)             # (batch, 7)
            pred   = torch.argmax(probs, dim=1)               # (batch,)
            conf   = probs.max(dim=1).values                  # (batch,)
        return pred, conf, probs

    def get_attention_weights(self, x):
        # extract attention weights for explainability
        # shows which sensor pairs the model focused on for this prediction
        self.eval()
        attention_weights = []

        def hook(module, input, output):
            # TransformerEncoderLayer does not expose attn weights directly
            # we register a hook on the self_attn submodule
            pass

        with torch.no_grad():
            embeddings = self.encoder(x)
            x_pos      = embeddings + self.transformer.pos_encoding
            # get attention from first transformer layer
            attn_output, attn_weights = self.transformer.transformer.layers[0].self_attn(
                x_pos, x_pos, x_pos, need_weights=True, average_attn_weights=True
            )
        return attn_weights   # (batch, 3, 3) -- attention between sensors


if __name__ == '__main__':
    batch_size    = 4
    embedding_dim = 128
    num_classes   = 7

    print('=== CrossModalTransformer ===')
    transformer = CrossModalTransformer(embedding_dim=embedding_dim, num_heads=4)
    x           = torch.randn(batch_size, 3, embedding_dim)
    out         = transformer(x)
    print(f'Input  : {x.shape}')
    print(f'Output : {out.shape}')   # expect (4, 128)
    n_params = sum(p.numel() for p in transformer.parameters())
    print(f'Params : {n_params:,}')

    print('\n=== JointFaultTransformer (no pre-trained weights) ===')
    model = JointFaultTransformer(
        embedding_dim   = embedding_dim,
        num_heads       = 4,
        num_layers      = 2,
        num_classes     = num_classes,
    )
    x   = torch.randn(batch_size, 3, 10000)
    out = model(x)
    print(f'Input  : {x.shape}')
    print(f'Output : {out.shape}')   # expect (4, 7)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Total params : {n_params:,}')

    # test predict method
    pred, conf, probs = model.predict(x)
    print(f'\nPredictions : {pred}')
    print(f'Confidence  : {conf}')
    print(f'Probs shape : {probs.shape}')

    # test attention weights
    attn = model.get_attention_weights(x)
    print(f'\nAttention weights shape: {attn.shape}')   # expect (4, 3, 3)
    print('All checks passed.')