import torch
import pandas as pd
import gc
import cv2

def empty_cached():
  gc.collect()
  torch.cuda.empty_cache()


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


def preprocessing(data, path):
  
  data['missed'] =  train.segmentation.notna()
  data['segm'] = ''

  for i in range(0, train.shape[0], 3):
      data.segm.iloc[i] = data.iloc[i:i+3].segmentation.to_numpy()
      data.missed.iloc[i] = data.iloc[i:i+3].missed.any()
  print(data.info())  

  data = train[::3]
  data.reset_index(drop = True, inplace = True)

  data = get_path(data, path)
  

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


def find_contour(img):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)[1]
    #thresh = cv2.erode(thresh, None, iterations=2)
    #thresh = cv2.dilate(thresh, None, iterations=2)

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)

    extreme_point = []
    extLeft = tuple(c[c[:,:,0].argmin()][0]); extreme_point.append(extLeft)
    extRight = tuple(c[c[:,:,0].argmax()][0]); extreme_point.append(extRight)
    extTop = tuple(c[c[:,:,1].argmin()][0]); extreme_point.append(extTop) 
    extBot = tuple(c[c[:,:,1].argmax()][0]); extreme_point.append(extBot)
    
    img_cnt = cv2.drawContours(img.copy(), c, -1, (0,255,255), 3)

    img_pnt = cv2.circle(img_cnt.copy(), extLeft, 5, (0, 0, 255), -1)
    img_pnt = cv2.circle(img_pnt, extRight, 5, (0, 255, 0), -1)
    img_pnt = cv2.circle(img_pnt, extTop, 5, (255, 0, 0), -1)
    img_pnt = cv2.circle(img_pnt, extBot, 5, (255, 255, 0), -1)
    
    return img_pnt, extreme_point


def crop_img(img, extreme_point):
    
    new_img = img[extreme_point[2][1]:extreme_point[3][1], extreme_point[0][0]:extreme_point[1][0]]
    
    return new_img


def remove_noise(img, extreme_point):

    new_img = np.zeros((img.shape[0],img.shape[1], img.shape[2]), dtype=np.uint8)
    
    new_img[extreme_point[2][1]:extreme_point[3][1], extreme_point[0][0]:extreme_point[1][0]] = img[extreme_point[2][1]:extreme_point[3][1], extreme_point[0][0]:extreme_point[1][0]]
    
    return new_img


