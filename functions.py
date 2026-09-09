import torch
import pandas as pd
import gc
import cv2

def empty_cached():
  gc.collect()
  torch.cuda.empty_cache()


def preprocessing_trainset(trainset, path):

  for i in range(0, data.shape[0], 3):
    trainset['segmentation'].iloc[i] = trainset['segmentation'].iloc[i:i+3].to_numpy()

  trainset = trainset[::3]
  trainset.reset_index(drop = True, inplace = True)
  
  trainset['path']     = path
  trainset['splitted'] = trainset['id'].agg(lambda x: x.split('_'))
  trainset['case']     = trainset['splitted'].agg(lambda x: str(x[0]))
  trainset['day']      = trainset['splitted'].agg(lambda x: str(x[1]))
  trainset['slice']    = trainset['splitted'].agg(lambda x: int(x[3].lstrip('0')))


  cases = trainset.case.value_counts(sort = False)
  cases_val, case_iter = cases.index, cases.values
  case_iter = np.insert(case_iter, 0, 0)
  case_iter = np.cumsum(case_iter)
  
  for i in range(1, case_iter.shape[0]):
      
      temp_data = trainset.loc[case_iter[i-1]:case_iter[i]-1]
  
      days = temp_data.day.value_counts(sort = False)
      day_val, day_iter = days.index, days.values
  
      day_iter = np.insert(day_iter, 0, 0)
      day_iter = np.cumsum(day_iter)
  
      for j in range(1, day_iter.shape[0]):
          
          first_idx = case_iter[i-1]+day_iter[j-1]
          second_idx = day_iter[j]+case_iter[i-1]-1
  
          val = cases_val[i-1]+'/'+cases_val[i-1]+'_'+day_val[j-1]+'/'+'scans/'
          
          trainset['path'].loc[first_idx] +=  val
  
          trainset['segmentation'].loc[first_idx] = data['segmentation'].loc[first_idx:second_idx].tolist()
  
          trainset.drop(data.loc[first_idx+1:second_idx].index, 
                 inplace = True)
  
    trainset.reset_index(inplace = True, drop = True)
    return trainset
    

def RLE_masking(rle_arr: np.array(str | None), 
                shape: (3, int, int, int)) -> torch.Tensor:
    """
    Ouput: np.array[3, Depth, Width, Height] of RLE masks
    for original(not transformed) 3D Object"
    """
    
    RLE = torch.zeros((shape[1], shape[0], shape[2], shape[3]))
    
    for j in range(shape[1]):
        for i in range(shape[0]):
            
            if pd.isna(rle_arr[j][i]):
                continue
    
            rle = list(map(int, rle_arr[j][i].split()))
            start_pixel, len_of_masking = rle[::2], rle[1::2]
            temp = torch.zeros((shape[2]*shape[3]))
            
            masking = torch.hstack(
                [torch.arange(start_pixel[i], start_pixel[i]+len_of_masking[i])
                                for i in range(len(start_pixel))]) 
            
            temp[masking] = 1 
            RLE[j][i] = temp.reshape((shape[2], shape[3]))
    
    RLE = RLE.permute(1, 0, 2, 3)
    return RLE


