import os
import re
import sys
import six
import random
import PIL
import lmdb
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from transforms import CVColorJitter, CVDeterioration, CVGeometry
import torchvision.transforms as transforms

MEAN_IMAGENET = torch.tensor([0.485, 0.456, 0.406])
STD_IMAGENET  = torch.tensor([0.229, 0.224, 0.225])

def get_dataloader(opt, dataset, batch_size, shuffle = False, mode = "label"):
    """
    Get dataloader for each dataset

    Parameters
    ----------
    opt: argparse.ArgumentParser().parse_args()
    dataset: torch.utils.data.Dataset
    batch_size: int
    shuffle: boolean

    Returns
    ----------
    data_loader: torch.utils.data.DataLoader
    """

    if mode == "raw":
        myAlignCollate = AlignCollateRaw(opt)
    else:
        myAlignCollate = AlignCollate(opt, mode)
    
    data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=opt.workers,
            collate_fn=myAlignCollate,
            pin_memory=False,
            drop_last=False,
        )
    return data_loader


def hierarchical_dataset(root, opt, mode="label", drop_data=[]):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f"dataset_root:    {root}\t dataset:"
    print(dataset_log)
    dataset_log += "\n"

    listdir = list()
    for dirpath, dirnames, filenames in os.walk(root + "/"):
        if not dirnames:
            # print(dirpath)
            flag = True
            for u in drop_data:
                if u in dirpath:
                    flag = False
                    break
            if flag == True:
                listdir.append(dirpath)

    listdir.sort()

    for dirpath in listdir:
        if mode == "raw":
            # load data without label
            dataset = LmdbDataset_raw(dirpath, opt)
        else:
            # load data with label
            dataset = LmdbDataset(dirpath, opt)
        sub_dataset_log = f"sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}"
        print(sub_dataset_log)
        dataset_log += f"{sub_dataset_log}\n"
        dataset_list.append(dataset)
    
    # concatenate many dataset
    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


class Pseudolabel_Dataset(Dataset):
    """
    Assign pseudo labels to data

    Parameters
    ----------
    unlabel_dataset: torch.utils.data.Dataset
    psudolabel_list: list(object) of pseudo labels
    """
    
    def __init__(self, unlabel_dataset, psudolabel_list):
        self.unlabel_dataset = unlabel_dataset
        self.psudolabel_list = psudolabel_list
        self.nSamples= len(self.psudolabel_list)

    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, index):
        label = self.psudolabel_list[index]
        img = self.unlabel_dataset[index]
        return (img, label)  


class AlignCollate(object):
    """ Transform data to the same format """
    def __init__(self, opt, mode = "label"):
        self.opt = opt
        # resize image
        if (mode == "adapt"):
            self.transform = Augment_tfs(opt)
        else:
            self.transform = Resize((opt.imgW, opt.imgH))
        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
 

    def __call__(self, batch):
        images, labels = zip(*batch)

        image_tensors = [self.normalize(self.totensor(self.transform(image))) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels
    
class AlignCollateRaw(object):
    """ Transform data to the same format """
    def __init__(self, opt):
        self.opt = opt
        # resize image
        self.transform = Resize((opt.imgW, opt.imgH))
        self.normalize = transforms.Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
        self.totensor  = transforms.ToTensor()

    def __call__(self, batch):
        images = batch

        image_tensors = [self.normalize(self.totensor(self.transform(image))) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors
    
    
class AlignCollateHDGE(object):
    """ Transform data to the same format """
    def __init__(self, opt, infer=False):
        self.opt = opt
        # For transforming the input image
        if infer == False:
            transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                transforms.Resize((opt.load_height,opt.load_width)),
                transforms.RandomCrop((opt.crop_height,opt.crop_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        else:
            transform = transforms.Compose(
                [transforms.Resize((opt.crop_height,opt.crop_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            
        self.transform = transform

    def __call__(self, batch):
        images = batch
        image_tensors = [self.transform(image) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors    


class LmdbDataset(Dataset):
    """ Load data from Lmdb file with label """
    def __init__(self, root, opt):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(
            root,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            print("cannot open lmdb from %s" % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get("num-samples".encode()))
            self.filtered_index_list = []
            for index in range(self.nSamples):
                index += 1  # lmdb starts with 1
                label_key = "label-%09d".encode() % index
                label = txn.get(label_key).decode("utf-8")

                # length filtering
                length_of_label = len(label)
                if length_of_label > opt.batch_max_length or length_of_label <= 0:
                    continue

                self.filtered_index_list.append(index)

            self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def _next_image(self, index):
        next_index = random.randint(0, len(self) - 1)
        return self.__getitem__(next_index)

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = "label-%09d".encode() % index
            label = txn.get(label_key).decode("utf-8")
            label = re.sub('[^0-9a-zA-Z]+', '', label)
            img_key = "image-%09d".encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img = PIL.Image.open(buf).convert("RGB")

            except IOError:
                print(f"Corrupted image for {index}")
                # make dummy image and dummy label for corrupted image.
                img = PIL.Image.new("RGB", (self.opt.imgW, self.opt.imgH))
                label = "[dummy_label]"

        return (img, label)


class LmdbDataset_raw(Dataset):
    """ Load data from Lmdb file without label """
    def __init__(self, root, opt):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(
            root,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            print("cannot open lmdb from %s" % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get("num-samples".encode()))
            self.index_list = [index + 1 for index in range(self.nSamples)]

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        index = self.index_list[index]

        with self.env.begin(write=False) as txn:
            img_key = "image-%09d".encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img = PIL.Image.open(buf).convert("RGB")

            except IOError:
                print(f"Corrupted image for {img_key}")
                # make dummy image for corrupted image.
                img = PIL.Image.new("RGB", (self.opt.imgW, self.opt.imgH))

        return img


class RandomCrop(object):
    """RandomCrop,
    RandomResizedCrop of PyTorch 1.6 and torchvision 0.7.0 work weird with scale 0.90-1.0.
    i.e. you can not always make 90%~100% cropped image scale 0.90-1.0, you will get central cropped image instead.
    so we made RandomCrop (keeping aspect ratio version) then use Resize.
    """

    def __init__(self, scale=[1, 1]):
        self.scale = scale

    def __call__(self, image):
        width, height = image.size
        crop_ratio = random.uniform(self.scale[0], self.scale[1])
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)

        x_start = random.randint(0, width - crop_width)
        y_start = random.randint(0, height - crop_height)
        image_crop = image.crop(
            (x_start, y_start, x_start + crop_width, y_start + crop_height)
        )
        return image_crop
    


class Resize(object):
    def __init__(self, size = (128,32)):
        # CAUTION: it should be (width, height). different from size of transforms.Resize (height, width)
        self.size = size
        self.img_w = size[0]
        self.img_h = size[1]

    def resize(self, img):
        return cv2.resize(img, (self.img_w, self.img_h), interpolation= cv2.INTER_CUBIC)

    def __call__(self, image):
        #image = self.resize(np.array(image))
        image = image.resize(self.size, PIL.Image.BICUBIC)
        return image

class Augment_tfs(object):
    """Augmentation from ABINet repo"""
    def __init__(self, opt):
        self.opt = opt
        self.Augment = transforms.Compose([
                    CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                    CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                    CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
        ])
        self.resize = Resize()
        print("Use Text_augment", self.Augment)

    def __call__(self, image):
        return self.resize(self.Augment(image))
