import os
import numpy as np
import torch

data_dir = './datasets'

class Dataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, transform=None):
    self.data_dir = data_dir
    self.transform = transform

    lst_data = os.listdir(self.data_dir)
    lst_label = [f for f in lst_data if f.startswith('label')]
    lst_input = [f for f in lst_data if f.startswith('input')]

    lst_label.sort()
    lst_input.sort()

    self.lst_label = lst_label
    self.lst_input = lst_input

  def __len__(self):
    return len(self.lst_label)

    
  def __getitem__(self, index):
    label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
    input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

    label = label/255.
    input = input/255.

    if label.ndim == 2:
      label = label[:,:,np.newaxis]
    if input.ndim == 2:
      input = input[:,:,np.newaxis]

    data = {'input':input, 'label':label}

    if self.transform:
      data = self.transform(data)
    
    return data
data_train = Dataset(data_dir=os.path.join(data_dir,'train'))

data = data_train.__getitem__(0)

input = data['input']
label = data['label']