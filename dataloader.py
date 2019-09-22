from __future__ import print_function, division

import os
import sys
import glob

from functools import partial

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor

import numpy as np
import random
import csv

from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from operator import itemgetter

import protonets
from protonets.data.base import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image

import json


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco      = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        # load_images and load_annotations defined below
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample
        # returns the transformed sample we want using its idx

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path       = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox'] # first four coordinates are the coordinates of the bbox
            annotation[0, 4]  = self.coco_label_to_label(a['category_id']) # last coordinate gives the object label
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        # used to be upper left corner (x,y), width (w) and height (h)
        # becomes the upper left corner, and bottom right corner
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    # coco labels are numbers from 1 to 90, there are 80 of them in total because there are 80 objects
    # this function returns a number between 0 to 79, the labels
    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    # inverse of coco_label_to_label above
    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        # there are 80 classes in COCO dataset
        return 80



def collater(data):
    # how to assemble the data for the batches: 
    # padding sequential data to max length of a batch

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img
        # add zeros to the right and bottom if image is smaller than max size of the batch

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
                    # add -1 to the right if less annotations than max number of annotations in the batch
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    # rescale image

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape
        # rows, columns, channels 

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""
    # flip images horizontally to augment the data

    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        # pretrained on ImageNet
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']

        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots}

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


coco_path = '/scratch/users/isarao/coco'

dataset_train = CocoDataset(coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Resizer()]))
        
map = {}
for key, value in dataset_train.classes.items(): 
    catIds = dataset_train.coco.getCatIds(catNms=[key])
    imgIds = dataset_train.coco.getImgIds(catIds=catIds)
    
    map[key] = imgIds

class_names = list(map.keys())

COCO_DATA_DIR  = os.path.join('/scratch/users/isarao/coco/images')
COCO_CACHE = { }

def load_image_path(key, out_field, d):
    img = Image.open(d[key])
    d[out_field] = img.copy()
    img.close()
    return d
    
def load_class_images(d):
    if d['class'] not in COCO_CACHE:
        image_dir = os.path.join(COCO_DATA_DIR, 'train2017')
        
        class_images = []
        for values in map[d['class']]: 
            class_images.append(values)
            
        class_images = sorted(class_images)
       
        image_ds = TransformDataset(ListDataset(class_images),
                                    compose([partial(convert_dict, 'image_id')]))

        loader = torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

        for sample in loader:
            COCO_CACHE[d['class']] = sample['image_id']
            break # only need one sample because batch size equal to dataset length

    return { 'class': d['class'], 'data': COCO_CACHE[d['class']] }

def extract_episode(n_sample, d):
    # data: N x C x H x W
    n_examples = len(d['data'])

    example_inds = torch.randperm(n_examples)[:n_sample]

    x = itemgetter(*example_inds)(d['data'])

    return {
        'class': d['class'],
        'x': x
    }

def load():

    ret = { }
    n_way = 2 # how to do a random number for the classes
    n_sample = 10
    n_episodes = 5
    
    transforms = [partial(convert_dict, 'class'),
                  load_class_images,
                  partial(extract_episode, n_sample)]
    
    transforms = compose(transforms)

    ds = TransformDataset(ListDataset(class_names), transforms)

    sampler = EpisodicBatchSampler(len(ds), n_way, n_episodes)

    # use num_workers=0, otherwise may receive duplicate episodes
    ret = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)

    return ret


def load_image(image_id):
    image_info = dataset_train.coco.loadImgs(image_id)[0]
    path       = os.path.join(coco_path, 'images', 'train2017', image_info['file_name'])
    img = skimage.io.imread(path)

    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    return img.astype(np.float32)/255.0

categories = dataset_train.coco.loadCats(dataset_train.coco.getCatIds())
categories.sort(key=lambda x: x['id'])
    
coco_labels_inverse = {}
classes = {}
for c in categories:
    coco_labels_inverse[c['id']] = len(classes)
    classes[c['name']] = len(classes)
    

def coco_label_to_label(coco_label):
    return coco_labels_inverse[coco_label]

def load_annotations(image_id):
    # get ground truth annotations
    annotations_ids = dataset_train.coco.getAnnIds(imgIds=image_id, iscrowd=False)
    annotations     = np.zeros((0, 5))

    # some images appear to miss annotations (like image with id 257034)
    if len(annotations_ids) == 0:
        return annotations

    # parse annotations
    coco_annotations = dataset_train.coco.loadAnns(annotations_ids)
    for idx, a in enumerate(coco_annotations):

        # some annotations have basically no width / height, skip them
        if a['bbox'][2] < 1 or a['bbox'][3] < 1:
            continue

        annotation        = np.zeros((1, 5))
        annotation[0, :4] = a['bbox']
        annotation[0, 4]  = coco_label_to_label(a['category_id'])
        annotations       = np.append(annotations, annotation, axis=0)

    # transform from [x, y, w, h] to [x1, y1, x2, y2]
    annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
    annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

    return annotations




