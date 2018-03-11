import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
from PIL import Image
from utils import DatasetFromFolder
from skimage.io import imsave


def get_cat_loader(batch_size):
    transforms = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
    ds64 = DatasetFromFolder("data/cats_bigger_than_64x64", transform=transforms)
    ds128 = DatasetFromFolder("data/cats_bigger_than_128x128", transform=transforms)
    ds_concat = ConcatDataset((ds64, ds128))
    data_loader = DataLoader(ds_concat, batch_size=batch_size, shuffle=True)
    return data_loader

if __name__ == '__main__':
    k = 0
    for img in data_loader:
        img_np = img.numpy()[0].swapaxes(0,1).swapaxes(1,2)
        print(img_np.shape)
        imsave(arr=img_np, fname="tmp/%i.png" % k)
        k += 1
        if k==10:
            break
