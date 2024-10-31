from ast import Tuple
import decimal
from numpy import ones
import torch
from torch import nn, tensor
import torch.nn.functional as F

from typing import Union, Tuple, Any

class PartialConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias = False,
        padding_mode: str = 'reflect',
        device = None,
        dtype = None
    ):
        super(PartialConv2d, self).__init__()
        ##################### 参数引入 #####################
        if kernel_size is not Tuple:
            self.kernel_size = (kernel_size, kernel_size)
        else:
        self.kernel_size = kernel_size
        # print(f"kernel_size is {kernel_size}")
        self.stride = stride
        self.bias: bool = bias
        ####################################################
    
    #################### 创建卷积核 ####################
        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,    # 在卷积之后手动添加偏置
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )
            # 设定一个其中参数全为1的卷积, 用于计算公式中的系数 sum(I)/sum(M)
        self.conv_1 = self.conv
        nn.init.constant_(self.conv_1.weight, 1)
    ####################################################
    def forward(self, in_x, in_mask):
        # print(f"origin shape is {in_x.shape}")
        # print(f"origin mask shape: {in_mask.shape}")
        ##################### 参数引入 #####################
        self.in_x = in_x
        self.in_mask = in_mask
        ####################################################
        
        ######################################### 部分卷积过程 #########################################
        # 1. 计算输入特征与掩膜的逐元素相乘 (X ⊙ M)
        self.calc_x = self.in_x * self.in_mask
        
        # 2. 卷积操作 (W^T · (X ⊙ M))
        self.out: torch.Tensor = self.conv(self.calc_x) # out1
        
        # 3. 计算系数矩阵 sum(I)/sum(M)
        with torch.no_grad():
            # print(f"self.conv_1 is {self.conv_1}")
            # print(f"self.conv is {self.conv}")
            self.sum_mask: torch.Tensor = self.conv_1(self.in_mask) # 利用一个其中参数全为1的卷积核与输入掩膜相乘, 算出每个像素的sum(M)
            
            # print(f"sum mask shape: {self.sum_mask.shape}")
            
            self.sum_I: int = self.kernel_size * self.kernel_size * 1# sum(I)是一个固定值, 从数值上与卷积核的窗口大小相等
            if (self.out.shape == self.sum_mask.shape):
                self.ratio: torch.Tensor = torch.where(
                    self.sum_mask<=0, 
                    torch.tensor(0.0, device=self.sum_mask.device), 
                    self.sum_I/self.sum_mask
                )
                # print(f'after:{self.ratio}')
            else:
                print("error!")
        # 4. 第二步中的卷积操作结果与第三步中系数矩阵的乘积, 即进行(W^T · (X ⊙ M)) · sum(I)/sum(M)
        self.out: torch.Tensor = self.out * self.ratio # out2
        
        if self.bias is True: #如果卷积核需要添加偏置：
            # 5. 创建一个偏置, 用于弥补卷积核中没有的偏置项, 即进行 (W^T · (X ⊙ M)) · sum(I)/sum(M) + b
            self.b = nn.Parameter(torch.zeros(list(self.out.shape)))  # 手动创建偏置参数
            self.out = self.out + self.b # 将偏置加入到结果中 # out3 # 返回值1
        ################################################################################################
        
        ######################################### 掩膜更新过程 #########################################
        self.updated_mask: torch.Tensor = torch.where((self.sum_mask!=0), torch.tensor(1.0, device=self.sum_mask.device), self.sum_mask).float() # # tensor != 0：生成一个与原张量形状相同的布尔张量，标记出哪些元素不是0。torch.tensor(1)：如果条件为真（即元素不为0），则将其设置为1。tensor：如果条件为假（即元素为0），则保持原样。# 返回值2
        # print(f"updated mask shape: {self.updated_mask.shape}")
        # print(f"result shape is {self.out.shape}")
        # print(f"-------------------------------------------------------------")
        ################################################################################################
        
        return self.out, self.updated_mask
        
if __name__ == '__main__':
    pc = PartialConv2d(in_channels=3, out_channels=3, kernel_size=(3,3), stride=1)
    in_x = torch.rand(size=(1,3,256,256)).float()
    in_mask = torch.randint(low=0, high=2, size=in_x.shape).float()
    # in_mask = torch.zeros_like(in_x)
    print(in_mask)

    
    out_x, out_mask = pc(in_x, in_mask)
    print(f'mask is {out_mask}')
    print(out_mask==1)