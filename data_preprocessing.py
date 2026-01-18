import numpy as np
import pandas as pd
import os

def get_path(data, path):
    data['path'] = path
    data['splitted'] = data.id.agg(lambda x: x.split('_'))
    data['case']     = data.splitted.agg(lambda x: str(x[0]))
    data['day']      = data.splitted.agg(lambda x: str(x[1]))
    data['slice']    = data.splitted.agg(lambda x: int(x[3].lstrip('0')))


    cases = data.case.value_counts(sort = False)
    cases_val, case_iter = cases.index, cases.values
    case_iter = np.insert(case_iter, 0, 0)
    case_iter = np.cumsum(case_iter)

    for i in range(1, case_iter.shape[0]):
    
        temp_data = data.iloc[case_iter[i-1]:case_iter[i]]
        data.path.iloc[case_iter[i-1]:case_iter[i]] +=  cases_val[i-1]

        days = temp_data.day.value_counts(sort = False)
        day_val, day_iter = days.index, days.values
        day_iter = np.insert(day_iter, 0, 0)
        day_iter = np.cumsum(day_iter)

        for j in range(1, day_iter.shape[0]):
        
            val = '/'+cases_val[i-1]+'_'+day_val[j-1]+'/'+'scans/'
            data.path.iloc[case_iter[i-1]+day_iter[j-1]:day_iter[j]+case_iter[i-1]] +=  val

            list_of_files = sorted(os.listdir(data.path.iloc[case_iter[i-1]+day_iter[j-1]]))

            data.path.iloc[case_iter[i-1]+day_iter[j-1]:day_iter[j]+case_iter[i-1]] +=  list_of_files

    return data



def preprocessing(data, column, path):
  
  data['missed'] =  data[column].notna()
  data['segm'] = ''

  for i in range(0, data.shape[0], 3):
      data.segm.iloc[i] = data.iloc[i:i+3].segmentation.to_numpy()
      data.missed.iloc[i] = data.iloc[i:i+3].missed.any()
      
  data = data[::3]
  data.reset_index(drop = True, inplace = True)

  data = get_path(data, path)

  return data
