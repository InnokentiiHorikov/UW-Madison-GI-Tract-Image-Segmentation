import numpy as np
import pandas as pd
from data_preprocessing import *
import warnings
import time
import cv2
import matplotlib.pyplot as plt
import os

warnings.filterwarnings("ignore")

path = 'files/train.csv'
path_to_processed_data = 'files/processed_data.csv'
path_to_data = 'files/Data/'

def main():

    if not os.path.isfile(path_to_processed_data):
        data = pd.read_csv(path)

        data = preprocessing(data, 'segmentation',
                              path_to_data)

        data.to_csv(path_to_processed_data)

    else:
        data = pd.read_csv(path_to_processed_data)
    

    fig, ax = plt.subplots()
    pf = data.path.iloc[0]
    ax.imshow(cv2.imread(pf, cv2.IMREAD_ANYDEPTH),cmap = 'gray')
    plt.show()
if __name__ == '__main__':
    main()


