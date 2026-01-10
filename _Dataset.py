import cv2


class ImageDataset(Dataset):
    def __init__(self, data, transform = None):
        
        super().__init__()
        
        self.data = data
        self.transform = transform

    
    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]    
        image = cv2.imread(item['path'], cv2.IMREAD_ANYDEPTH)/65355

        if ((image.shape[0] + image.shape[1] < 720)):
            image = cv2.copyMakeBorder(image, 0, 360 - image.shape[0], 
                                       360-image.shape[1], 0,
                                       cv2.BORDER_CONSTANT)            
  
        masks = torch.zeros((3, 360, 360), dtype = torch.half)

        
        segmentation = item['segm']
        image = torch.tensor(image, dtype = torch.half)

        for i in range(3):
            masks[i, :, :] = RLE_masking(segmentation[i])
        
        return {'image': image, 'masks': masks}
