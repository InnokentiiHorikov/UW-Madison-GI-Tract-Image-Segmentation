from torch import nn
from torch.nn.functional import pad


class Right(nn.Module):
    def __init__(self, 
               in_chan, out_chan):
        super(Right, self).__init__()
        
        self.conv_1 = nn.Conv3d(in_chan, out_chan, kernel_size = 3, padding = 'same')

        self.conv_2 = nn.Conv3d(out_chan, out_chan, kernel_size = 3, padding = 'same')
        self.BN = nn.BatchNorm3d(out_chan)
        
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
                nn.MaxPool3d(2),
                Right(in_chan, out_chan))

    def forward(self, _input):
            return self.down_step(_input)


class Up(nn.Module):
    def __init__(self, 
               in_chan, out_chan):
        super(Up, self).__init__() 

        self.up = nn.ConvTranspose3d(in_chan, in_chan // 2, kernel_size=2)
        self.conv = Right(in_chan, out_chan)    
        

    def forward(self, input1, input2):
            
        x1 = self.up(input1)
        
        diffZ = input2.size()[2] - x1.size()[2]
        diffY = input2.size()[3] - x1.size()[3]
        diffX = input2.size()[4] - x1.size()[4]

        x1 = pad(x1, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([input2, x1], dim=1)
        return self.conv(x)




class Out(nn.Module):
    def __init__(self, 
               in_chan, out_chan):
        super(Out, self).__init__() 

        self.conv = nn.Conv3d(in_chan, out_chan, kernel_size = 1)  

    def forward(self, x):
        return self.conv(x)

