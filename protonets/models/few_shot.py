import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from protonets.models import register_model

from .utils import euclidean_dist

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()
        
        self.encoder = encoder

    def loss(self, sample):
        xs = Variable(sample['xs']) # support
        #print('xs: ', xs.size()) 
        #xs:  torch.Size([5, 5, 1, 28, 28])
        xq = Variable(sample['xq']) # query
        #print('xq:', xq.size()) 
        #xq: torch.Size([5, 15, 1, 28, 28])

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
        
        if torch.eq(y_hat, target_inds.squeeze()).float().mean() != 1: 
            diff = torch.eq(y_hat, target_inds.squeeze())
            index = (diff == 0).nonzero()
            for i in range(0, index.size(0)):
                if i == 0: 
                    misclassified = xq[index.data[i].data[0], index.data[i].data[1], :, :, :]
                else: 
                    misclassified = torch.cat((misclassified, xq[index.data[i].data[0], index.data[i].data[1], :, :, :]), 0)
        
        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

    def pred(self, sample): 
        xs = Variable(sample['xs']) # support
        #print(xs.size())
        xq = Variable(sample['xq']) # query
        #print(xq.size())
        class_name = sample['class']

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        _, y_hat = log_p_y.max(2)

        # return {
        #     'sample':xq, 
        #     'pred': y_hat, 
        #     'orig': target_inds.squeeze()
        # }

        #print('y_hat', y_hat)
        #print('labels', target_inds.squeeze())
        #print('data', xq)
        
        #misclassified = torch.zeros(0)
        misclassified = [] 
        wrong_image = []
        correct_label = []
        wrong_label = []
        
        if torch.eq(y_hat, target_inds.squeeze()).float().mean() != 1: 
            #print('class names are: ', class_name)
            diff = torch.eq(y_hat, target_inds.squeeze())
            #print('y_hat: ', y_hat)
            #print('y: ', target_inds.squeeze())
            index = (diff == 0).nonzero()
            for i in range(0, index.size(0)):
                # if i == 0: 
                #     misclassified = xq[index.data[i].data[0], index.data[i].data[1], :, :, :]
                # else: 
                #     misclassified = torch.cat((misclassified, xq[index.data[i].data[0], index.data[i].data[1], :, :, :]), 0)
                predicted_class_num = y_hat[index.data[i].data[0], index.data[i].data[1]]
                misclassified.append(xq[index.data[i].data[0], index.data[i].data[1], :, :, :])
                # row is a class, and column is an example
                # example of the predicted class that is wrong
                wrong_image.append(xs[predicted_class_num, 0, :, :, :])
                correct_label.append(class_name[index.data[i].data[0]])
                #print('correct label is: ', class_name[index.data[i].data[0]])
                #print('correct label is: ', correct_label)
                wrong_label.append(class_name[predicted_class_num])
                #print('predicted label is: ', class_name[predicted_class_num])
                #print('predicted label is: ', wrong_label)

        # might have to convert numpy to tensor
        #return torch.cat((target_inds.squeeze(), y_hat), 0)
        return misclassified, wrong_image, correct_label, wrong_label



@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )

    return Protonet(encoder)
