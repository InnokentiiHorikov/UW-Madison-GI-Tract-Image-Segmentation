import torch
import pandas as pd


def get_path(data, path):
  train['path'] = path
  train['splitted'] = train.id.agg(lambda x: x.split('_'))
  train['case']     = train.splitted.agg(lambda x: str(x[0]))
  train['day']      = train.splitted.agg(lambda x: str(x[1]))
  train['slice']    = train.splitted.agg(lambda x: int(x[3].lstrip('0')))


  cases = train.case.value_counts(sort = False)
  cases_val, case_iter = cases.index, cases.values
  case_iter = np.insert(case_iter, 0, 0)
  case_iter = np.cumsum(case_iter)

  for i in range(1, case_iter.shape[0]):
    
      temp_data = train.iloc[case_iter[i-1]:case_iter[i]]
      train.path.iloc[case_iter[i-1]:case_iter[i]] +=  cases_val[i-1]

      days = temp_data.day.value_counts(sort = False)
      day_val, day_iter = days.index, days.values
      day_iter = np.insert(day_iter, 0, 0)
      day_iter = np.cumsum(day_iter)

      for j in range(1, day_iter.shape[0]):
        
          val = '/'+cases_val[i-1]+'_'+day_val[j-1]+'/'+'scans/'
          train.path.iloc[case_iter[i-1]+day_iter[j-1]:day_iter[j]+case_iter[i-1]] +=  val

          list_of_files = sorted(os.listdir(train.path.iloc[case_iter[i-1]+day_iter[j-1]]))

          train.path.iloc[case_iter[i-1]+day_iter[j-1]:day_iter[j]+case_iter[i-1]] +=  list_of_files

  return data


def preprocessing(data):
  
  data['missed'] =  train.segmentation.notna()
  data['segm'] = ''

  for i in range(0, train.shape[0], 3):
      data.segm.iloc[i] = data.iloc[i:i+3].segmentation.to_numpy()
      data.missed.iloc[i] = data.iloc[i:i+3].missed.any()
  print(train.info())  

  train = train[::3]
  train.reset_index(drop = True, inplace = True)

def RLE_masking(rle, shape = (360, 360)):
    
    if pd.isna(rle):
        return torch.zeros(shape[0], shape[1])

    size_of_image = shape[0]*shape[1]

    rle = list(map(int, rle.split()))
        
    start_pixel, len_of_masking = rle[::2], rle[1::2]
    mask = torch.zeros(size_of_image)
    
    masking = torch.hstack([torch.arange(start_pixel[i], start_pixel[i]+len_of_masking[i] + 1) 
                        for i in range(len(start_pixel))]) - 1

    mask[masking] = 1
    mask = torch.reshape(mask, (shape[0], shape[1]))
    
    return mask
