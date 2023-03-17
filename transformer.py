import torch
import numpy as np
from torchvision import transforms
from dataloader import Dataset
import os

data_dir = './datasets'

class ToTensor(object):
  def __call__(self, data):
      label, input = data['label'],data['input']
      label = label.transpose((2, 0, 1)).astype(np.float32)
      input = input.transpose((2, 0, 1)).astype(np.float32)

      data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}
      return data
  
class Normalization(object):
  def __init__(self, mean=0.5, std=0.5):
    self.mean = mean
    self.std = std
  
  def __call__(self, data):
    label, input= data['label'], data['input']

    input = (input - self.mean) / self.std

    data = {'label':label, 'input':input}
    return data
  
class RandomFlip(object):
  def __call__(self, data):
    label, input = data['label'], data['input']

    if np.random.rand() > 0.5:
      label = np.fliplr(label)
      input = np.fliplr(input)

      
    if np.random.rand() > 0.5:
      label = np.flipud(label)
      input = np.flipud(input)

    data = {'label' : label, 'input' : input}
    return data

transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
dataset_train = Dataset(data_dir=os.path.join(data_dir,'train'), transform=transform)

data = dataset_train.__getitem__(0)

input = data['input']
label = data['label']