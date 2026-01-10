from torch import nn
from torch.functional import pad

class Right(nn.Module):
    def __init__(self, 
               in_chan, out_chan):
        
        super(Right, self).__init__()
        
        self.conv_1 = nn.Conv2d(in_chan, out_chan, kernel_size = 3)
        self.conv_2 = nn.Conv2d(out_chan, out_chan, kernel_size = 3)
        
        self.BN = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()

    
    def forward(self, _input):

            x = self.conv_1(_input)
            x = self.BN(x)
            x = self.relu(x)
            
            x = self.conv_2(x)
            x = self.BN(x)
            output = self.relu(x)

            return output



class Down(nn.Module):
    def __init__(self, 
               in_chan, out_chan):
        
        super(Down, self).__init__() 

        self.down_step = nn.Sequential(
                nn.MaxPool2d(2),
                Right(in_chan, out_chan)
        )

    def forward(self, _input):
        
            return self.down_step(_input)


class Up(nn.Module):
    def __init__(self, 
               in_chan, out_chan):
        
        super(Up, self).__init__() 

        self.up = nn.ConvTranspose2d(in_chan, in_chan // 2, kernel_size=2)
        self.conv = Right(in_chan, out_chan)    
        

    def forward(self, input1, input2):
            
        x1 = self.up(input1)
        
        diffY = input2.size()[2] - x1.size()[2]
        diffX = input2.size()[3] - x1.size()[3]

        x1 = pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([input2, x1], dim=1)
        return self.conv(x)




class Out(nn.Module):
    def __init__(self, 
               in_chan, out_chan):
        
        super(Out, self).__init__() 

        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size = 1)  

    def forward(self, x):
        return self.conv(x)




class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

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

        self.log = nn.Softmax()

    def forward(self, x):
        x = x.view(x.shape[0], 1, 360,360)


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

        
        return self.log(logits)

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
