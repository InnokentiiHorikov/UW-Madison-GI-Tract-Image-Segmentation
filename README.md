First of all, I want to create model from scratch and understand the principes of a creating multiclass segmentation. As a result, a model results would be not great as should be. If you noticed a significant issues in the model or data preparation/analysis, please message me about it. I watched several notebooks from other users, and they used pre-trained model 

```python
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
        '''
        list_of_files = np.repeat(list_of_files, 
                                  np.ones(len(list_of_files), dtype = np.int8)*3)
        '''
        train.path.iloc[case_iter[i-1]+day_iter[j-1]:day_iter[j]+case_iter[i-1]] +=  list_of_files
        
```
