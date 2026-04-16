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
    mat = scipy.io.loadmat(filepath)
    signals = np.stack([
        mat['A'].squeeze(),
        mat['B'].squeeze(),
        mat['C'].squeeze(),
    ], axis=0).astype(np.float32)
    return signals


def get_split_files(data_dir):
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


def preprocess_and_cache(files, cache_dir, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    total_windows = 0
    for i, filepath in enumerate(files):
        info       = parse_filename(filepath)
        label      = info['class_label']
        cache_path = cache_dir / (filepath.stem + '.pt')
        if cache_path.exists():
            data           = torch.load(cache_path, weights_only=True)
            total_windows += data['windows'].shape[0]
            continue
        signals = load_mat_signals(filepath)
        n       = signals.shape[1]
        starts  = list(range(0, n - window_size + 1, stride))
        windows = np.stack([signals[:, s:s+window_size] for s in starts])
        torch.save({
            'windows': torch.from_numpy(windows),
            'label'  : label,
            'file'   : filepath.name,
        }, cache_path)
        total_windows += len(starts)
        if (i + 1) % 20 == 0 or (i + 1) == len(files):
            print(f'  cached {i+1}/{len(files)} files  ({total_windows:,} windows so far)')
    print(f'Cache complete: {total_windows:,} windows in {cache_dir}')
    return total_windows


class ORIONDataset(Dataset):

    def __init__(self, files, transform=None, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE, cache_dir=None):
        self.files       = files
        self.transform   = transform
        self.window_size = window_size
        self.stride      = stride
        self.cache_dir   = Path(cache_dir) if cache_dir else None
        self.cache_data  = {}

        if self.cache_dir and self.cache_dir.exists():
            self._build_index_from_cache()
        else:
            self._build_index()

    def _build_index_from_cache(self):
        # load ALL cache files into RAM at startup
        # fast on Mac which has enough RAM
        # __getitem__ then just indexes into in-memory tensors -- very fast
        self.index = []

        for filepath in self.files:
            cache_path = self.cache_dir / (filepath.stem + '.pt')
            if not cache_path.exists():
                # fallback to raw loading if cache missing
                info      = parse_filename(filepath)
                label     = info['class_label']
                mat       = scipy.io.loadmat(filepath, variable_names=['A'])
                n_samples = mat['A'].shape[0]
                for s in range(0, n_samples - self.window_size + 1, self.stride):
                    self.index.append(('raw', filepath, label, s))
                continue

            # load entire .pt file into RAM
            data = torch.load(cache_path, weights_only=True)
            key  = filepath.stem
            self.cache_data[key] = data

            label     = data['label']
            n_windows = data['windows'].shape[0]
            for i in range(n_windows):
                self.index.append(('cached', key, label, i))

        print(f'Dataset built from cache: {len(self.index):,} windows from {len(self.files)} files')

    def _build_index(self):
        self.index = []
        for filepath in self.files:
            info      = parse_filename(filepath)
            label     = info['class_label']
            mat       = scipy.io.loadmat(filepath, variable_names=['A'])
            n_samples = mat['A'].shape[0]
            for s in range(0, n_samples - self.window_size + 1, self.stride):
                self.index.append((filepath, label, s))
        print(f'Dataset built: {len(self.index):,} windows from {len(self.files)} files')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index[idx]

        if entry[0] == 'cached':
            _, key, label, window_idx = entry
            window = self.cache_data[key]['windows'][window_idx]
        else:
            filepath, label, start = entry[0], entry[1], entry[2] if len(entry) == 3 else entry[3]
            signals = load_mat_signals(filepath)
            window  = torch.from_numpy(signals[:, start:start + self.window_size])

        if self.transform:
            window = self.transform(window.float())

        return window.float(), label


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    DATA_DIR  = Path('data/raw/ORION_AE_acoustic_emission_multisensor_datasets_bolts_loosening')
    CACHE_DIR = Path('data/processed/cache')

    train_files, test_files = get_split_files(DATA_DIR)

    print('\nCaching train files...')
    preprocess_and_cache(train_files, CACHE_DIR)

    print('\nCaching test files...')
    preprocess_and_cache(test_files, CACHE_DIR)

    print('\nTesting cached dataset...')
    dataset       = ORIONDataset(train_files[:2], cache_dir=CACHE_DIR)
    window, label = dataset[0]
    print(f'Window shape : {window.shape}')
    print(f'Label        : {label}')

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    x, y   = next(iter(loader))
    print(f'Batch x      : {x.shape}')
    print(f'Batch y      : {y}')
    print('\nAll checks passed.')