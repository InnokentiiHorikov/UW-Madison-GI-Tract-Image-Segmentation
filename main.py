import pandas as pd
import numpy as np
import preproc

path = '/kaggle/input/uw-madison-gi-tract-image-segmentation/train.csv'

if __name__ == "__main__":
  data = pd.read_csv(path)
  data = preproc.preprocessing(data)

  
  
