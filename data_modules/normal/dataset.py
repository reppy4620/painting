import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_image


class NormalDataset(Dataset):

    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

        self.gray_kernel = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float)[None, :, None, None]
        self.blur = T.GaussianBlur(7, sigma=1.4)
        self.blur2 = T.GaussianBlur(7, sigma=1.4 * 1.6)
        self.normalize = T.Normalize([0.5]*3, [0.5]*3)

    def __len__(self):
        return len(self.img_paths)

    def _xdog(self, color, p=100, phi=15):
        color = F.conv2d(color, self.gray_kernel)
        g1 = self.blur(color)
        g2 = self.blur2(color)
        out = (1+p) * g1 - p * g2
        out /= out.max()
        out = 1 + torch.tanh(phi * out)
        out = out.clamp(0, 1)
        return out

    def _extract_sketch(self, color):
        x = F.conv2d(color, self.gray_kernel)
        dilated = F.max_pool2d(x, 3, 1, 1)
        x = 1.0 - torch.abs(x-dilated)
        return x

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        color = read_image(str(img_path)) / 255
        if self.transform:
            color = self.transform(color)
        sketch = self._xdog(color.unsqueeze(0))[0]
        color = self.normalize(color)
        return sketch, color
