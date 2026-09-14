import tqdm
import torchio as tio
import torch
import torchmetrics

def train_test_model(epochs: int,
                     model: torch.nn.Module,
                     loss_fn: torch.nn.Module,
                     metrics: torchmetrics.Metric,
                     device: torch.device,
                     optimizer: torch.optim.Optimizer,
                     train_patches_loader: tio.SubjectsLoader,
                     valid_patches_loader: tio.SubjectsLoader
               ) -> None:
                 
    for i in range(epochs):
      print(f"Epoch {i+1}")
      
      train_tqdm = tqdm.tqdm(train_patches_loader)
      model.train()
      
      for batch in train_tqdm:
          
          image, masks = batch['image'][tio.DATA], batch['segmentation'][tio.DATA]
          image, masks = image.to(device), masks.to(device) 
          #N, C, H, W, D -> N, C, D, H, W
          image, masks = image.permute(0, 1, 4, 2, 3), masks.permute(0, 1, 4, 2, 3)
  
          optimizer.zero_grad() 
          model.zero_grad()
          
          output = model(image)
          L = loss_fn(output, masks)
          L.backward()
  
          
          output = (F.sigmoid(output) > 0.3).float() 
          M = metrics(output.to(torch.int64), masks.to(torch.int64))
          
          optimizer.step()   
          train_tqdm.set_description(f"Train loss: {L.item()}  Metrics loss: {M}") 
          train_tqdm.refresh() 
          
      model.eval()
      valid_metrics = torch.tensor([0, 0, 0], dtype = torch.float).to(device)
      val_loss = []
      
      test_tqdm = tqdm.tqdm(valid_patches_loader)
      
      with torch.no_grad():
          for batch in test_tqdm:
              image, masks = batch['one_image'][tio.DATA], batch['a_segmentation'][tio.DATA]
              image, masks = image.to(device), masks.to(device) 
              image, masks = image.permute(0, 1, 4, 2, 3), masks.permute(0, 1, 4, 2, 3)
                  
              output = model(image)
  
              L = loss_fn(output, masks)
              val_loss.append(L.item())
              
              output = torch.round(F.sigmoid(output))
              M = metrics(output.to(torch.int64), masks.to(torch.int64))
              
              valid_metrics = torch.vstack((valid_metrics, M))
              
      print(f"Epoch {i}, Loss: {np.mean(val_loss)} Metrics: {torch.mean(valid_metrics[1:, :], axis = 0)}")

  torch.save(model.state_dict(), 'model.pth')
