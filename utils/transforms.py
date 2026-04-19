import torch
import torch.nn.functional as F


class Normalise:
    # scale each window to [-1, 1] per channel
    # removes effect of different sensor sensitivities
    # done per window not per file -- each 2ms window normalised independently
    def __call__(self, x):
        # x shape: (3, 10000)
        mins  = x.min(dim=1, keepdim=True).values
        maxs  = x.max(dim=1, keepdim=True).values
        denom = (maxs - mins).clamp(min=1e-8)   # avoid divide by zero on silent windows
        return 2 * (x - mins) / denom - 1


class AddNoise:
    # add gaussian noise -- used for SimCLR augmentation in Phase 1
    # std=0.01 means noise is 1% of the normalised signal range
    # std=0.05 means 5% noise -- larger, used for domain augmentation
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std


class RandomScale:
    # randomly scale amplitude by factor in [1-scale, 1+scale]
    # scale=0.1 -- small variation for SimCLR
    # scale=0.4 -- large variation for domain augmentation (up to 40%)
    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, x):
        factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.scale
        return x * factor


class RandomDCOffset:
    # adds random constant offset per sensor channel
    # simulates different noise floor and baseline voltage between campaigns
    # each campaign has slightly different electronic baseline due to remounting
    def __init__(self, offset=0.1):
        self.offset = offset

    def __call__(self, x):
        # x shape: (3, 10000) -- different offset per sensor channel
        offsets = (torch.rand(3, 1) * 2 - 1) * self.offset
        return x + offsets


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
        resized  = F.interpolate(
            cropped.unsqueeze(0), size=n, mode='linear', align_corners=False
        ).squeeze(0)
        return resized


class Compose:
    # chain multiple transforms together
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


# Phase 2 supervised training -- normalise only, no augmentation, deterministic
supervised_transform = Compose([
    Normalise(),
])

# Phase 1 SimCLR -- small augmentations to create positive pairs
# call twice on the same window to get two different augmented views
simclr_transform = Compose([
    Normalise(),
    AddNoise(std=0.01),
    RandomScale(scale=0.1),
    RandomCrop(crop_ratio=0.1),
])

# domain augmentation -- large augmentations to simulate sensor coupling
# variations between campaigns:
#   AddNoise(0.05)       -- different electronic noise floor (5% not 1%)
#   RandomScale(0.4)     -- different sensor sensitivity (up to 40% variation)
#   RandomDCOffset(0.1)  -- different baseline voltage per channel
#   RandomCrop(0.1)      -- different window alignment
# used during Phase 2 training when doing campaign-wise generalisation
domain_augment_transform = Compose([
    Normalise(),
    AddNoise(std=0.02),       # reduced from 0.05
    RandomScale(scale=0.2),   # reduced from 0.4
    RandomDCOffset(offset=0.05),  # reduced from 0.1
    RandomCrop(crop_ratio=0.05),  # reduced from 0.1
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

    x_offset = RandomDCOffset(offset=0.1)(x_norm)
    print('\nAfter RandomDCOffset:')
    print('  range : [{:.3f}, {:.3f}]'.format(x_offset.min().item(), x_offset.max().item()))

    x_crop = RandomCrop(crop_ratio=0.1)(x_norm)
    print('\nAfter RandomCrop:')
    print('  shape :', x_crop.shape)

    x_sup = supervised_transform(x)
    print('\nSupervised transform output range    : [{:.3f}, {:.3f}]'.format(
        x_sup.min().item(), x_sup.max().item()))

    x_sim = simclr_transform(x)
    print('SimCLR transform output range        : [{:.3f}, {:.3f}]'.format(
        x_sim.min().item(), x_sim.max().item()))

    x_dom = domain_augment_transform(x)
    print('Domain augment transform output range: [{:.3f}, {:.3f}]'.format(
        x_dom.min().item(), x_dom.max().item()))

    print('\nAll transforms OK.')