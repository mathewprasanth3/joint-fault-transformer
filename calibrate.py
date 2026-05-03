import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent))

from utils.dataset import ORIONDataset, get_split_files, NUM_CLASSES
from utils.transforms import supervised_transform, domain_augment_transform
from models.transformer.model import JointFaultTransformer


def evaluate_campaign_f(
    data_dir,
    model_path    = 'models/transformer/weights/best_model_bcde.pt',
    embedding_dim = 128,
    num_heads     = 4,
    num_layers    = 2,
    batch_size    = 64,
    num_workers   = 0,
    cache_dir     = 'data/processed/cache',
):
    # baseline cross-campaign eval -- no fine-tuning
    # tests model trained on B,C,D,E directly on all of campaign F
    from sklearn.metrics import classification_report, confusion_matrix

    print('Step 1: baseline cross-campaign evaluation (no fine-tuning)')
    print('  train: B, C, D, E   test: F')

    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    _, f_files = get_split_files(data_dir)
    dataset    = ORIONDataset(f_files, transform=supervised_transform, cache_dir=cache_dir)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f'Campaign F windows: {len(dataset):,}')

    model = JointFaultTransformer(
        embedding_dim = embedding_dim,
        num_heads     = num_heads,
        num_layers    = num_layers,
        num_classes   = NUM_CLASSES,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.float().to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy   = (all_preds == all_labels).mean()

    print(f'\nBaseline cross-campaign accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')

    class_names = ['05cNm', '10cNm', '20cNm', '30cNm', '40cNm', '50cNm', '60cNm']
    print('\nClassification report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print('Confusion matrix:')
    print(confusion_matrix(all_labels, all_preds))

    return accuracy


def calibrate_and_evaluate(
    data_dir,
    model_path    = 'models/transformer/weights/best_model_bcde.pt',
    save_path     = 'models/transformer/weights/calibrated_model.pt',
    embedding_dim = 128,
    num_heads     = 4,
    num_layers    = 2,
    batch_size    = 32,
    num_epochs    = 30,
    lr            = 1e-5,        # small lr -- full model fine-tuning, not retraining
    cal_ratio     = 0.15,        # 15% of F for calibration, 85% for test
    num_workers   = 0,
    cache_dir     = 'data/processed/cache',
):
    # full model calibration fine-tuning on campaign F
    # all layers train -- encoder, transformer, and classifier
    # small lr (1e-5) prevents catastrophic forgetting of learned features
    from sklearn.metrics import classification_report, confusion_matrix

    print('\nStep 2: full model calibration fine-tuning')
    print(f'  calibration set : {int(cal_ratio*100)}% of F')
    print(f'  test set        : {int((1-cal_ratio)*100)}% of F')
    print(f'  trainable       : all layers (encoder + transformer + classifier)')
    print(f'  lr              : {lr} (small to avoid catastrophic forgetting)')

    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    _, f_files = get_split_files(data_dir)

    # two dataset objects over same files -- different transforms
    cal_dataset  = ORIONDataset(f_files, transform=domain_augment_transform, cache_dir=cache_dir)
    test_dataset = ORIONDataset(f_files, transform=supervised_transform,     cache_dir=cache_dir)

    total     = len(cal_dataset)
    cal_size  = int(total * cal_ratio)
    test_size = total - cal_size

    # same seed on both splits so cal/test indices match
    cal_subset,  _ = random_split(cal_dataset,  [cal_size, test_size], generator=torch.Generator().manual_seed(99))
    _, test_subset = random_split(test_dataset, [cal_size, test_size], generator=torch.Generator().manual_seed(99))

    print(f'Calibration windows : {len(cal_subset):,}')
    print(f'Test windows        : {len(test_subset):,}')

    cal_loader  = DataLoader(cal_subset,  batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = JointFaultTransformer(
        embedding_dim = embedding_dim,
        num_heads     = num_heads,
        num_layers    = num_layers,
        num_classes   = NUM_CLASSES,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # all layers trainable
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f'Trainable params: {trainable:,} / {total_p:,} ({trainable/total_p*100:.1f}%)')

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=num_epochs)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    best_loss        = float('inf')
    patience_counter = 0
    early_stop       = 10

    print(f'\n{"Epoch":>6} | {"Loss":>8} | {"Acc":>8}')
    print('-' * 28)

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in cal_loader:
            x, y   = x.float().to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
            correct    += (logits.argmax(dim=1) == y).sum().item()
            total      += y.size(0)

        avg_loss = total_loss / len(cal_loader)
        avg_acc  = correct / total
        scheduler.step()

        print(f'{epoch+1:>6} | {avg_loss:>8.4f} | {avg_acc:>8.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop:
            print(f'Early stopping at epoch {epoch+1}')
            break

    print(f'\nBest calibration loss: {best_loss:.4f}')
    print(f'Saved: {save_path}')

    # eval on held-out campaign F test set
    print('\nFinal eval on held-out campaign F test set')
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x.float().to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy   = (all_preds == all_labels).mean()

    print(f'Calibrated accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')

    class_names = ['05cNm', '10cNm', '20cNm', '30cNm', '40cNm', '50cNm', '60cNm']
    print('\nClassification report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print('Confusion matrix:')
    print(confusion_matrix(all_labels, all_preds))

    return accuracy


if __name__ == '__main__':
    DATA_DIR = Path('data/raw/ORION_AE_acoustic_emission_multisensor_datasets_bolts_loosening')

    baseline_acc = evaluate_campaign_f(
        data_dir   = DATA_DIR,
        model_path = 'models/transformer/weights/best_model_bcde.pt',
        cache_dir  = 'data/processed/cache',
    )

    calibrated_acc = calibrate_and_evaluate(
        data_dir   = DATA_DIR,
        model_path = 'models/transformer/weights/best_model_bcde.pt',
        save_path  = 'models/transformer/weights/calibrated_model.pt',
        cal_ratio  = 0.30,
        num_epochs = 50,
        lr         = 1e-5,
        cache_dir  = 'data/processed/cache',
    )

    print('\nSummary')
    print(f'  random split accuracy (B+C+D+E+F mixed) :  87.00%')
    print(f'  cross-campaign baseline (no calibration) :  {baseline_acc*100:.2f}%')
    print(f'  after full model calibration (15% of F)  :  {calibrated_acc*100:.2f}%')
    print(f'  accuracy recovered                       : +{(calibrated_acc - baseline_acc)*100:.2f}%')