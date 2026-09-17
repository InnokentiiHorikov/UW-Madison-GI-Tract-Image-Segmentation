from UNet3d_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, size):
        super(UNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.size = size 

        self.inc = Right(n_channels, out_chan = 64)
        
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        self.outc = Out(64, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        
        self.outc = torch.utils.checkpoint(self.outc)

class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.2, 
                 beta: float = 1.0, eps: float = 1e-6):
        """
        Custom Tversky Loss Module.
        alpha: controls penalty for false positives.
        beta: controls penalty for false negatives.
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert raw logits to probabilities
        outputs = F.sigmoid(outputs).to(torch.float32)
        
        # Flatten batch, spatial dimensions
        outputs = outputs.reshape(outputs.shape[0], outputs.shape[1], -1)
        targets = targets.reshape(targets.shape[0], targets.shape[1], -1)
        
        # Calculate True Positives, False Positives, False Negatives
        true_pos = (outputs * targets).sum(dim=2)
        false_pos = ((1 - targets) * outputs).sum(dim=2)
        false_neg = (targets * (1 - outputs)).sum(dim=2)
        
        # Calculate Tversky Index
        tversky_index = (true_pos + self.eps) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.eps
        )
        
        return (1.0 - tversky_index.mean()).to(torch.float32)



def init_weights(m):
    
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
    
        
    elif isinstance(m, (nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1.0)  
        nn.init.constant_(m.bias, 0.01)   

