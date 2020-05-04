import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ProjectDown(nn.Module):
    def __init__(self,ip_size):
        _,n_ip_channel,h,w = ip_size
        super().__init__()
        if n_ip_channel < 64 :
            n_op_channel = 64
        elif n_ip_channel < 128 :
            n_op_channel = 128
        else:
            n_op_channel = 256
        ops = []
        ops.append(nn.Conv2d(n_ip_channel,n_op_channel, kernel_size=1))
        ops.append(nn.Conv2d(n_op_channel,64, kernel_size=1))
        ops.append(nn.ReLU())
        ops.append(nn.BatchNorm2d(64))
        ops.append(nn.AdaptiveAvgPool2d(4))

        self.op = nn.Sequential(*ops)

    def forward(self, x):
        x = self.op(x) #[?,64,4,4]
        x = x.view(-1,64,16)
        return x

class ProjectUp(nn.Module):
    def __init__(self,final_size):
        super().__init__()
        _,n_ip_channel,h,w = final_size
        # input is (-1,64,16)
        # input is (-1,64,4,4)

        self.op = nn.Sequential(
            nn.AdaptiveAvgPool2d(h),
            nn.Conv2d(64,128,1),
            nn.Conv2d(128,n_ip_channel,1)
        )

    def forward(self,x):
        x = x.view(-1,64,4,4)
        return self.op(x)

if __name__ == "__main__":
    x = torch.randn(32,96,3,3)
    model = ProjectDown(x.size())
    y=model.forward(x)
    print(f"Y : {y.size()}")

    model = ProjectUp(x.size())
    y=model.forward(y)
    print(f"Y : {y.size()}")





