import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.dataset import ORIONDataset, get_split_files, NUM_CLASSES
from utils.transforms import supervised_transform
from models.transformer.model import JointFaultTransformer


def train_phase2(
    data_dir,
    encoder_weights  = 'models/encoder/weights/simclr_encoder.pt',
    save_path        = 'models/transformer/weights/best_model_bcde.pt',
    embedding_dim    = 128,
    num_heads        = 4,
    num_layers       = 2,
    batch_size       = 64,
    num_epochs       = 75,
    lr               = 1e-4,
    dropout          = 0.1,
    train_split      = 0.8,
    val_split        = 0.1,
    num_workers      = 0,
    freeze_encoder   = False,
    cache_dir        = 'data/processed/cache',
):
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # train on B, C, D, E only -- campaign F held out for cross-campaign eval
    train_files, _ = get_split_files(data_dir)

    print(f'Total files : {len(train_files)} (campaigns B, C, D, E only)')

    full_dataset = ORIONDataset(train_files, transform=supervised_transform, cache_dir=cache_dir)

    # random 80/10/10 split with fixed seed for reproducibility
    total      = len(full_dataset)
    test_size  = int(total * (1 - train_split))
    val_size   = int((total - test_size) * val_split)
    train_size = total - test_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f'Train windows : {train_size:,}')
    print(f'Val windows   : {val_size:,}')
    print(f'Test windows  : {test_size:,}')
    print(f'Batches/epoch : {len(train_loader)}')

    encoder_path = encoder_weights if Path(encoder_weights).exists() else None
    if encoder_path is None:
        print('WARNING: no pre-trained encoder found -- training from random weights')

    model = JointFaultTransformer(
        embedding_dim   = embedding_dim,
        num_heads       = num_heads,
        num_layers      = num_layers,
        num_classes     = NUM_CLASSES,
        dropout         = dropout,
        encoder_weights = encoder_path,
    ).to(device)

    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print('Encoder frozen -- only transformer and classifier will train')

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', patience=5, factor=0.5
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    best_val_loss       = float('inf')
    patience_counter    = 0
    early_stop_patience = 15

    for epoch in range(num_epochs):

        model.train()
        train_loss    = 0.0
        train_correct = 0
        train_total   = 0

        for x, y in train_loader:
            x, y = x.float().to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss    += loss.item()
            preds          = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total   += y.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc      = train_correct / train_total

        model.eval()
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y   = x.float().to(device), y.to(device)
                logits = model(x)
                loss   = criterion(logits, y)
                val_loss    += loss.item()
                preds        = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total   += y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc      = val_correct / val_total

        scheduler.step(avg_val_loss)

        print(f'Epoch {epoch+1:3d}/{num_epochs} | '
              f'train loss {avg_train_loss:.4f} acc {train_acc:.4f} | '
              f'val loss {avg_val_loss:.4f} acc {val_acc:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f'  saved best model (val loss {best_val_loss:.4f})')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f'Early stopping at epoch {epoch+1} -- val loss not improving')
            break

    print(f'\nPhase 2 complete. Best val loss: {best_val_loss:.4f}')
    print(f'Model saved: {save_path}')


def evaluate(
    data_dir,
    model_path    = 'models/transformer/weights/best_model_bcde.pt',
    embedding_dim = 128,
    num_heads     = 4,
    num_layers    = 2,
    batch_size    = 64,
    num_workers   = 0,
    cache_dir     = 'data/processed/cache',
):
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np

    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')

    # rebuild same split with same seed -- B, C, D, E only
    train_files, _ = get_split_files(data_dir)
    full_dataset   = ORIONDataset(train_files, transform=supervised_transform, cache_dir=cache_dir)

    total      = len(full_dataset)
    test_size  = int(total * 0.2)
    val_size   = int((total - test_size) * 0.1)
    train_size = total - test_size - val_size

    _, _, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f'Test windows: {len(test_dataset):,}')

    model = JointFaultTransformer(
        embedding_dim = embedding_dim,
        num_heads     = num_heads,
        num_layers    = num_layers,
        num_classes   = NUM_CLASSES,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x      = x.float().to(device)
            logits = model(x)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    print(f'\nTest accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')

    class_names = ['05cNm', '10cNm', '20cNm', '30cNm', '40cNm', '50cNm', '60cNm']
    print('\nClassification report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print('Confusion matrix:')
    print(confusion_matrix(all_labels, all_preds))


if __name__ == '__main__':
    DATA_DIR = Path('data/raw/ORION_AE_acoustic_emission_multisensor_datasets_bolts_loosening')

    train_phase2(
        data_dir        = DATA_DIR,
        encoder_weights = 'models/encoder/weights/simclr_encoder.pt',
        save_path       = 'models/transformer/weights/best_model_bcde.pt',
        embedding_dim   = 128,
        num_heads       = 4,
        num_layers      = 2,
        batch_size      = 64,
        num_epochs      = 75,
        lr              = 1e-4,
        dropout         = 0.1,
        train_split     = 0.8,
        val_split       = 0.1,
        freeze_encoder  = False,
        cache_dir       = 'data/processed/cache',
    )