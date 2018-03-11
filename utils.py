import os, gzip, torch
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset

from PIL import Image

def load_mnist(dataset):
    data_dir = os.path.join("./data", dataset)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY).astype(np.int)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1

    X = X.transpose(0, 3, 1, 2) / 255.
    # y_vec = y_vec.transpose(0, 3, 1, 2)

    X = torch.from_numpy(X).type(torch.FloatTensor)
    y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    return X, y_vec

def load_celebA(dir, transform, batch_size, shuffle):
    # transform = transforms.Compose([
    #     transforms.CenterCrop(160),
    #     transform.Scale(64)
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])

    # data_dir = 'data/celebA'  # this path depends on your computer
    dset = datasets.ImageFolder(dir, transform)
    data_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle)

    return data_loader


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%03d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


class DatasetFromFolder(Dataset):
    """
    Specify specific folders to load images from.

    Notes
    -----
    Courtesy of:
    https://github.com/togheppi/CycleGAN/blob/master/dataset.py
    With some extra modifications done by me.
    """
    def __init__(self, image_dir, images=None, transform=None, append_label=None):
        """
        Parameters
        ----------
        image_dir: directory where the images are located
        images: a list of images you want instead. If set to `None` then it gets all
          images in the directory specified by `image_dir`.
        transform:
        resize_scale: a tuple (w,h) denoting the resolution to resize to. Note that if
          `crop_size` is also defined, this will happen before the cropping.
        crop_size: a tuple (w,h) denoting the size of random crops to be made from
          this image.
        fliplr: enable left/right flip augmentation?
        append_label: if an int is provided, then `__getitem__` will return not just
          the image x, but (x,y), where y denotes the label. This means that this
          iterator could also be used for classifiers.
        """
        super(DatasetFromFolder, self).__init__()
        #self.input_path = os.path.join(image_dir, subfolder)
        self.input_path = image_dir
        if images == None:
            self.image_filenames = [x for x in sorted(os.listdir(self.input_path))]
        else:
            if type(images) != set:
                images = set(images)
            self.image_filenames = [ os.path.join(image_dir, fname) for fname in images ]
        self.transform = transform
        self.append_label = append_label
    def __getitem__(self, index):
        # Load Image
        img_fn = os.path.join(self.input_path, self.image_filenames[index])
        img = Image.open(img_fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.append_label is not None:
            yy = np.asarray([self.append_label])
            return img, yy
        else:
            return img
    def __len__(self):
        return len(self.image_filenames)

def get_cat_loader(batch_size):
    trs = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5])
    ])
    dss = tuple([ DatasetFromFolder("data2/CAT_%s" % elem, transform=trs) for elem in ["01","02","03","04","05","06"] ])

    ds_concat = ConcatDataset( (dss[0], dss[1]) )
    ds_concat = ConcatDataset( (ds_concat, dss[2]) )
    ds_concat = ConcatDataset( (ds_concat, dss[3]) )
    ds_concat = ConcatDataset( (ds_concat, dss[4]) )
    ds_concat = ConcatDataset( (ds_concat, dss[5]) )    
    
    data_loader = DataLoader(ds_concat, batch_size=batch_size, shuffle=True)
    return data_loader

def get_mat_loader(batch_size):
    X = np.load("data2/Target_rsem_gene_fpkm.npy")
    X = torch.from_numpy(X)
    y = torch.from_numpy(np.ones(X.shape[0],))
    ds = TensorDataset(X, y)
    data_loader = DataLoader(ds, batch_size=batch_size)
    return data_loader

