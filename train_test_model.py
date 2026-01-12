from _Dataset import *
import tqdm

def train_test_model(data, model,
                     loss_fn, optimizer, metrics, 
                     epochs, batch_size):
  dataset = ImageDataset(train)
  trainset, validset = random_split(dataset, [0.9, 0.1])
  train_dataloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
  valid_dataloader = DataLoader(validset, shuffle=True, batch_size=batch_size)
  
  for i in range(epochs):
    print(f"Epoch {i}")
    
    train_tqdm = tqdm.tqdm(train_dataloader)
    model.train()
    
    for batch in train_tqdm:
        image, masks = batch['image'], batch['masks']
        masks = masks[:, :, :352, :352]
        image, masks = image.to(device), masks.to(device) 
    
        optimizer.zero_grad() 
        model.zero_grad()
        
        output = model(image)
        L = loss_fn(output, masks)
        output, masks = output.cpu(), masks.cpu()
        
        M = metrics(3, output, masks)
 

        L.backward() 
        nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)

        optimizer.step()  
        train_tqdm.set_description(f"Train loss: {L.item()}  Metrics loss: {M[0]}-{M[1]}-{M[2]}")
        train_tqdm.refresh() 


    model.eval()
    valid_metrics = torch.tensor([0, 0, 0], dtype = torch.float)
    val_loss = []
    
    test_tqdm = tqdm.tqdm(valid_dataloader)
    with torch.no_grad():
        for batch in test_tqdm:
            image, masks = batch['image'], batch['masks']
            masks = masks[:, :, :352, :352]
            image, masks = image.to(device), masks.to(device) 
            
            output = model(image)
            L = loss_fn(output, masks)
            val_loss.append(L.item())
            
            output, masks = output.cpu(), masks.cpu()
            M = metrics(3, output, masks)
            
            valid_metrics = torch.vstack((valid_metrics, M))

    print(f"Epoch {i}, Loss: {np.mean(val_loss)} Metrics: {torch.mean(valid_metrics[1:, :], axis = 0)}")
                            
