from tqdm import tqdm

import torch

from protonets.utils import filter_opt
from protonets.models import get_model

def load(opt):
    model_opt = filter_opt(opt, 'model')
    model_name = model_opt['model_name']

    del model_opt['model_name']

    return get_model(model_name, model_opt)

def evaluate(model, data_loader, meters, desc=None):
    model.eval()

    for field,meter in meters.items():
        # initialize the values in meters to 0
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    for sample in data_loader:
        _, output = model.loss(sample)
        for field, meter in meters.items():
            meter.add(output[field])

    return meters

def misclassifications(model, data_loader):
    model.eval() #set to evaluation mode
    
    # dummy = 0
    # for sample in data_loader:
    #     misclassified = model.pred(sample)
    #     if len(list(misclassified.size())) > 1: 
    #         if dummy == 0: 
    #             misclassifications = misclassified
    #             dummy = 1
    #         else: 
    #             misclassifications = torch.cat((misclassifications, misclassified), 0)
    #         print(misclassifications.size())
    
    misclassifications = []
    wrong_images = []
    correct_labels = []
    wrong_labels = []
    
    for sample in data_loader:
        misclassified, wrong_image, correct_label, wrong_label = model.pred(sample)
        misclassifications += misclassified
        wrong_images += wrong_image
        correct_labels += correct_label
        wrong_labels += wrong_label
    
    #print(wrong_labels)
    #print(correct_labels)

    return misclassifications, wrong_images, correct_labels, wrong_labels

    