import pandas as pd
import numpy as np
import preprocessing
import torch
from torch import nn 
from model/train_test_model import train_test_model
from model/UNet3D import *


path = '/kaggle/input/uw-madison-gi-tract-image-segmentation/train.csv'
path_to_MRI = '/kaggle/input/uw-madison-gi-tract-image-segmentation/images/'
path_to_nib = '/kaggle/working/'

device = ( "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
patch_size = (64, 64, 64)
epochs = 10


model = UNet(n_channels = 1, n_classes = 3, size = patch_size).to(dtype=torch.float32, device='cuda')
loss_fn = TverskyLoss() 
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,weight_decay=1e-3) 
metrics = GeneralizedDiceScore(num_classes=3, per_class=True).to(device)


def main():
    data = pd.read_csv(path)
    data = preprocessing.preprocessing_data(data, path_to_MRI)
    
    train_obj_dir, train_mask_dir, valid_obj_dir, valid_mask_dir =  preprocessing.to_create_paths(path_to_nib)

    tensor_to_gz(data = trainset, file_dir = (train_obj_dir, train_mask_dir),
             dtype = dtype, compress = None)

    tensor_to_gz(data = validset, file_dir = (valid_obj_dir, valid_mask_dir),
             dtype = dtype, compress = None)

    train_loader = creating_patch_loader(train_obj_dir, train_mask_dir, patch_size)
    valid_loader = creating_patch_loader(valid_obj_dir, valid_mask_dir, patch_size)
    model.apply(init_weights)
    
    train_test_model(data, model, 
                   loss_fn,optimizer, metrics,
                   epochs, batch_size)

    train_test_model(epochs = epochs, model = model, 
                     loss_fn = loss_fn, metrics = metrics, 
                     device = device, optimizer = optimizer, 
                     train_patches_loader = train_loader,
                     valid_loader = valid_loader)


if __name__ == "__main__":
    main()  


  
