import torch
import numpy as np


class Normalise:
    # scale each window to [-1, 1] per channel
    # removes effect of different sensor sensitivities
    # done per window not per file — each 2ms window normalised independently
    def __call__(self, x):
        # x shape: (3, 10000)
        mins  = x.min(dim=1, keepdim=True).values
        maxs  = x.max(dim=1, keepdim=True).values
        denom = (maxs - mins).clamp(min=1e-8)  # avoid divide by zero on silent windows
        return 2 * (x - mins) / denom - 1


class AddNoise:
    # add small gaussian noise — used for SimCLR augmentation in Phase 1
    # std=0.01 means noise is 1% of the normalised signal range
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std


class RandomScale:
    # randomly scale amplitude by a factor in [1-scale, 1+scale]
    # used for SimCLR augmentation — same event, slightly different amplitude
    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, x):
        factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.scale
        return x * factor


class RandomCrop:
    # randomly crop a sub-window and resize back to original length
    # forces model to be robust to which part of the 2ms window it sees
    def __init__(self, crop_ratio=0.1):
        self.crop_ratio = crop_ratio

    def __call__(self, x):
        # x shape: (3, 10000)
        n        = x.shape[1]
        crop_len = int(n * self.crop_ratio)
        start    = torch.randint(0, crop_len + 1, (1,)).item()
        end      = n - torch.randint(0, crop_len + 1, (1,)).item()
        cropped  = x[:, start:end]
        # resize back to original length using linear interpolation
        resized  = torch.nn.functional.interpolate(
            cropped.unsqueeze(0), size=n, mode='linear', align_corners=False
        ).squeeze(0)
        return resized


class Compose:
    # chain multiple transforms together — same as torchvision.transforms.Compose
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


# ready-made transform sets for each training phase

# Phase 2 supervised training — normalise only, no augmentation
supervised_transform = Compose([
    Normalise(),
])

# Phase 1 SimCLR — two augmented views of the same window
# call this twice on the same window to get a positive pair
simclr_transform = Compose([
    Normalise(),
    AddNoise(std=0.01),
    RandomScale(scale=0.1),
    RandomCrop(crop_ratio=0.1),
])


if __name__ == '__main__':
    # test all transforms on a random (3, 10000) tensor
    x = torch.randn(3, 10000)

    print('Input shape :', x.shape)
    print('Input range : [{:.3f}, {:.3f}]'.format(x.min().item(), x.max().item()))

    x_norm = Normalise()(x)
    print('\nAfter Normalise:')
    print('  range : [{:.3f}, {:.3f}]'.format(x_norm.min().item(), x_norm.max().item()))

    x_noise = AddNoise(std=0.01)(x_norm)
    print('\nAfter AddNoise:')
    print('  std diff : {:.6f}'.format((x_noise - x_norm).std().item()))

    x_scale = RandomScale(scale=0.1)(x_norm)
    print('\nAfter RandomScale:')
    print('  range : [{:.3f}, {:.3f}]'.format(x_scale.min().item(), x_scale.max().item()))

    x_crop = RandomCrop(crop_ratio=0.1)(x_norm)
    print('\nAfter RandomCrop:')
    print('  shape :', x_crop.shape)

    x_sup = supervised_transform(x)
    print('\nSupervised transform output range: [{:.3f}, {:.3f}]'.format(
        x_sup.min().item(), x_sup.max().item()))

    x_sim = simclr_transform(x)
    print('SimCLR transform output range   : [{:.3f}, {:.3f}]'.format(
        x_sim.min().item(), x_sim.max().item()))

    print('\nAll transforms OK.')