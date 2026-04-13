import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.dataset import ORIONDataset, get_split_files
from models.encoder.model import TripleEncoder


# fast vectorised SimCLR augmentation -- operates on whole batch at once
# much faster than looping window by window
def simclr_augment_batch(x):
    # x shape: (batch, 3, 10000)
    # normalise per channel per window
    mins  = x.min(dim=2, keepdim=True).values
    maxs  = x.max(dim=2, keepdim=True).values
    denom = (maxs - mins).clamp(min=1e-8)
    x     = 2 * (x - mins) / denom - 1
    # add gaussian noise
    x = x + torch.randn_like(x) * 0.01
    # random amplitude scale per window
    scale = 1.0 + (torch.rand(x.shape[0], 1, 1) * 2 - 1) * 0.1
    x     = x * scale
    return x


class NTXentLoss(nn.Module):

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # z1, z2 shape: (batch, embedding_dim)
        batch_size = z1.shape[0]

        # normalise to unit sphere
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # all pairwise similarities
        z   = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature

        # mask self similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim  = sim.masked_fill(mask, float('-inf'))

        # positive pairs are at positions (i, i+batch) and (i+batch, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(z.device)

        return F.cross_entropy(sim, labels)


class ProjectionHead(nn.Module):

    def __init__(self, embedding_dim=128, projection_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, x):
        # x shape: (batch, 3, embedding_dim)
        b, s, d = x.shape
        out     = self.net(x.reshape(b * s, d))
        return out.reshape(b, s, -1)


class SimCLRModel(nn.Module):

    def __init__(self, embedding_dim=128, projection_dim=64):
        super().__init__()
        self.encoder   = TripleEncoder(embedding_dim=embedding_dim)
        self.projector = ProjectionHead(embedding_dim, projection_dim)

    def forward(self, x):
        return self.projector(self.encoder(x))


def train_simclr(
    data_dir,
    save_path      = 'models/encoder/weights/simclr_encoder.pt',
    embedding_dim  = 128,
    projection_dim = 64,
    batch_size     = 64,
    num_epochs     = 50,
    lr             = 3e-4,
    temperature    = 0.5,
    num_workers    = 0,
    max_files      = None,
):
    # use MPS on Mac Apple Silicon, CUDA on GPU, otherwise CPU
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_files, _ = get_split_files(data_dir)

    if max_files:
        train_files = train_files[:max_files]
        print(f'Using {max_files} files for quick test')

    dataset = ORIONDataset(train_files, transform=None)
    loader  = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        drop_last   = True,
    )
    print(f'Train windows : {len(dataset):,}')
    print(f'Batches/epoch : {len(loader)}')

    model     = SimCLRModel(embedding_dim, projection_dim).to(device)
    criterion = NTXentLoss(temperature=temperature)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=num_epochs)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for x, _ in loader:
            x = x.float()

            # two different augmented views of the same batch
            x1 = simclr_augment_batch(x.clone()).to(device)
            x2 = simclr_augment_batch(x.clone()).to(device)

            # forward pass
            z1 = model(x1)   # (batch, 3, projection_dim)
            z2 = model(x2)

            # flatten sensors for NT-Xent: (batch*3, projection_dim)
            loss = criterion(
                z1.reshape(-1, projection_dim),
                z2.reshape(-1, projection_dim)
            )

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            n_batches  += 1

            if n_batches % 50 == 0:
                print(f'  epoch {epoch+1} | batch {n_batches}/{len(loader)} | loss {total_loss/n_batches:.4f}')

        avg_loss = total_loss / n_batches
        scheduler.step()
        print(f'Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.encoder.state_dict(), save_path)
            print(f'  saved encoder weights')

    print(f'\nPhase 1 complete. Best loss: {best_loss:.4f}')
    print(f'Encoder saved: {save_path}')


if __name__ == '__main__':
    DATA_DIR = Path('data/raw/ORION_AE_acoustic_emission_multisensor_datasets_bolts_loosening')

    train_simclr(
        data_dir       = DATA_DIR,
        save_path      = 'models/encoder/weights/simclr_encoder.pt',
        embedding_dim  = 128,
        projection_dim = 64,
        batch_size     = 64,
        num_epochs     = 50,
        lr             = 3e-4,
        temperature    = 0.5,
        max_files      = None,
    )