import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageNMaskDataset(Dataset):
    def __init__(self, root, transforms_image=None, transforms_mask=None, unaligned=False, mode='train'):
        self.transform_image = transforms.Compose(transforms_image)
        self.transform_mask = transforms.Compose(transforms_mask)
        self.unaligned = unaligned

        #img = Image(os.path.join(self.root, self.elements[index].rstrip()))
        self.files_A = sorted(glob.glob(os.path.join(root, 'A') + '/*.*'))
        self.files_M = sorted(glob.glob(os.path.join(root, 'B') + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, 'C') + '/*.*'))


    def __getitem__(self, index):
        item_A = self.transform_image(Image.open(self.files_A[index % len(self.files_A)]))
        item_M = self.transform_mask(Image.open(self.files_M[index % len(self.files_M)]).convert("L"))
        if self.unaligned:
            item_B = self.transform_image(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform_image(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B, 'M' : item_M}
        #return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B),  len(self.files_M))
        #return max(len(self.files_A), len(self.files_B))
