import re
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


WINDOW_SIZE   = 10_000
WINDOW_STRIDE = 5_000
FS            = 5_000_000

TRAIN_CAMPAIGNS = ['B', 'C', 'D', 'E']
TEST_CAMPAIGNS  = ['F']

# 0-indexed for PyTorch CrossEntropyLoss
TORQUE_TO_CLASS = {
    '05': 0,
    '10': 1,
    '20': 2,
    '30': 3,
    '40': 4,
    '50': 5,
    '60': 6,
}

TORQUE_TO_NM = {
    '05': '0.05 Nm (nearly free)',
    '10': '0.10 Nm (very loose)',
    '20': '0.20 Nm (significantly loose)',
    '30': '0.30 Nm (half nominal)',
    '40': '0.40 Nm (noticeable preload loss)',
    '50': '0.50 Nm (slightly reduced preload)',
    '60': '0.60 Nm (fully tight)',
}

NUM_CLASSES = 7


def parse_filename(filepath):
    # extract torque level and campaign from filename
    # e.g. salves_out_05cNm_B_Fs5MHz_... -> torque='05', campaign='B'
    name  = Path(filepath).name
    match = re.search(r'_(\d{2})cNm_([BCDEF])_', name)
    if not match:
        raise ValueError(f'Could not parse filename: {name}')
    torque_str = match.group(1)
    campaign   = match.group(2)
    return {
        'filename'   : name,
        'torque_str' : torque_str,
        'class_label': TORQUE_TO_CLASS[torque_str],
        'campaign'   : campaign,
        'torque_nm'  : TORQUE_TO_NM[torque_str],
    }


def load_mat_signals(filepath):
    # load one .mat file and return AE channels A, B, C as (3, N) float32
    # channel D is the vibrometer — excluded, validation only
    mat = scipy.io.loadmat(filepath)
    signals = np.stack([
        mat['A'].squeeze(),
        mat['B'].squeeze(),
        mat['C'].squeeze(),
    ], axis=0).astype(np.float32)
    return signals


def get_split_files(data_dir):
    # split all .mat files into train (B,C,D,E) and test (F) by campaign
    data_dir    = Path(data_dir)
    all_files   = sorted(data_dir.rglob('*.mat'))
    train_files = []
    test_files  = []
    for f in all_files:
        info = parse_filename(f)
        if info['campaign'] in TRAIN_CAMPAIGNS:
            train_files.append(f)
        elif info['campaign'] in TEST_CAMPAIGNS:
            test_files.append(f)
    print(f'Train files : {len(train_files)}  (campaigns B, C, D, E)')
    print(f'Test files  : {len(test_files)}   (campaign F — held out)')
    return train_files, test_files


class ORIONDataset(Dataset):

    def __init__(self, files, transform=None, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
        self.files       = files
        self.transform   = transform
        self.window_size = window_size
        self.stride      = stride
        self._build_index()

    def _build_index(self):
        # build a flat list of (filepath, label, window_start) for every window
        # across all files — lets DataLoader shuffle freely across files
        self.index = []
        for filepath in self.files:
            info      = parse_filename(filepath)
            label     = info['class_label']
            # load only channel A to get sample count — avoids loading full file
            mat       = scipy.io.loadmat(filepath, variable_names=['A'])
            n_samples = mat['A'].shape[0]
            for s in range(0, n_samples - self.window_size + 1, self.stride):
                self.index.append((filepath, label, s))
        print(f'Dataset built: {len(self.index):,} windows from {len(self.files)} files')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        filepath, label, start = self.index[idx]
        signals = load_mat_signals(filepath)
        window  = signals[:, start : start + self.window_size]  # (3, window_size)
        window  = torch.from_numpy(window)
        if self.transform:
            window = self.transform(window)
        return window, label


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    DATA_DIR = Path('../data/raw/ORION_AE_acoustic_emission_multisensor_datasets_bolts_loosening')

    train_files, test_files = get_split_files(DATA_DIR)

    # test with first 2 files only for speed
    dataset       = ORIONDataset(train_files[:2])
    window, label = dataset[0]
    print(f'Window shape : {window.shape}')
    print(f'Label        : {label}')

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    x, y   = next(iter(loader))
    print(f'Batch x      : {x.shape}')
    print(f'Batch y      : {y}')