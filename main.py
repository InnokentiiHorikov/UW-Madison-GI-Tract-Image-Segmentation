import pandas as pd
import numpy as np
import preproc
import torch
from torch import nn 
from torchmetrics.segmentation import GeneralizedDiceScore


path = '/kaggle/input/uw-madison-gi-tract-image-segmentation/train.csv'

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
batch_size = 12

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,weight_decay=1e-2) 
loss_fn = nn.BCEWithLogitsLoss() 
metrics = GeneralizedDiceScore(num_classes=3, per_class=True).to(device)

if __name__ == "__main__":
  data = pd.read_csv(path)
  data = preproc.preprocessing(data)

  
  
