import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import dataset_train, load_image, load_annotations, load, CocoDataset, collater, Resizer, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval
import csv_eval

#assert torch.__version__.split('.')[1] == '4'

sys.path = [''] + sys.path

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):

	parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
	parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

	parser = parser.parse_args(args)

	# Create the data loaders
	if parser.dataset == 'coco':

		if parser.coco_path is None:
			raise ValueError('Must provide --coco_path when training on COCO,')

		#dataset_train = CocoDataset(parser.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
		#dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))

		dataloader_train = load()

	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	#sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
	#dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

	#if dataset_val is not None:
	#		sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
	#	dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

	# Create the model
	# returns [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]
	if parser.depth == 18:
		retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 34:
		retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 50:
		retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 101:
		retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
	elif parser.depth == 152:
		retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
	else:
		raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')		

	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()
	
	retinanet = torch.nn.DataParallel(retinanet).cuda()

	retinanet.training = True

	optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

	loss_hist = collections.deque(maxlen=500)

	retinanet.train()
	# Freeze batch norm layers
	retinanet.module.freeze_bn()

	print('Num training images: {}'.format(len(dataset_train)))

	for epoch_num in range(parser.epochs):

		retinanet.train()
		retinanet.module.freeze_bn()
		
		epoch_loss = []
		
		# ~ engine in prototypical network
		for iter_num, data_temp in enumerate(dataloader_train): #iterates through the episodes
			try:
				optimizer.zero_grad()

				#print('size of data')
				#print(data.items()) 
				# data is a dictionary with keys: img, annot and scale 
				#print(data['img'])
				#print(data['img'].size()) 
				#(batch size (2), channels (3), width and height of image) (padded by 0 so every image in the batch has the same dimension)
				#print(data['annot'])
				#print(data['annot'].size())
				#(batch size (2), maximum number of annotations per image in the batch, coordinates + class id (5))
				# annotations are padded by -1 so every image in the batch has the same number of annotations
				#print(data['scale'])
				# vector of size 2 (size of batch) with the scale of the image

				# same for when using anchors: take the mean excluding values of -1

				classes_ids = dataset_train.classes
				relevant_ids = [classes_ids[x] for x in data_temp['class']]

				sample = [] 
				normalizer = Normalizer()
				resizer = Resizer()

				for i in range(len(data_temp['x'])): 
				    for j in range(len(data_temp['x'][0])): 
				        idx = data_temp['x'][i][j].item()
				        img = load_image(idx)
				        annots = load_annotations(idx)
				        # only keep annotations for the conisidered classes
				        annots = annots[np.isin(annots[:,4], relevant_ids)]
				        temp = {'img': img, 'annot': annots}
				        sample.append(resizer(normalizer(temp)))

				data = collater(sample)
				#print(data)
				# now the data is still a dictionary with keys: img, annot and scale 
				#print(data['img'].size())
				#print('test initial image')
				#print(data['img'].sum())
				#(number of images ~ batch size (=(n_support + n_query)*n_way), channels (=3), width, height)
				#print(data['annot'].size())
				#(number of images, max number of annotations in those images, coordinates & class (=5))
				#print(len(data['scale']))
				# list of length number of images, containing the scale for each
				#sys.exit()

				# need to change classification loss, and format of regression loss to accept the new form of the batch

				classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

				classification_loss = classification_loss.mean()
				regression_loss = regression_loss.mean()

				loss = classification_loss + regression_loss
				
				if bool(loss == 0):
					continue

				loss.backward()

				torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

				optimizer.step()

				loss_hist.append(float(loss))

				epoch_loss.append(float(loss))

				print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
				
				del classification_loss
				del regression_loss
			except Exception as e:
				print(e)
				continue

		# if parser.dataset == 'coco':

		# 	print('Evaluating dataset')

		# 	coco_eval.evaluate_coco(dataset_val, retinanet)


		
		scheduler.step(np.mean(epoch_loss))	

		#torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

	retinanet.eval()

	torch.save(retinanet, 'model_final.pt'.format(epoch_num))

if __name__ == '__main__':
 main()
