"""
该文件中包含一个工具函数, 用于完成以下任务：

我有一个深度检测的结果, 其数值类型是一个张量(Tensor). 我将该张量中数值不为4的分量全部改成0, 以达到掩膜的效果, 此时该数据还是一个张量(Tensor), 该文件中的函数实现了将这个带有黑色掩膜张量(Tensor)可视化的操作 

"""

from ast import List
import torch
import numpy as np
from PIL import Image

def get_mask_tensor(depth_tensor: torch.Tensor, t: int, mode='tensor'):
    """
    一个用于获取带有掩膜的深度检测张量(Tensor)的函数. 与t相同的数值将被设置为1, 否则被设置为0

    depth_tensor参数：需要可视化的深度预测张量(Tensor), 可以在cpu上, 也可以在gpu上;

    t参数: 需要掩膜的数值, 例如, 如果t=4, 则将深度检测张量中数值不为4的分量全部改成0, 其余部分修改为1

    mode参数：用于控制返回值类型, 可以为numpy或tensor, 默认为tensor

    返回：一个掩膜数组, 返回值类型与str参数保持一致
    """
    depth_tensor = depth_tensor.cpu()
    # 将数值不为4的分量全部改成0
    masked_tensor = torch.where(depth_tensor == t, torch.tensor(1, dtype=torch.uint8), torch.tensor(0, dtype=torch.uint8))

    mode = mode.upper()

    if mode == 'TENSOR':
        return masked_tensor
    elif mode == 'NUMPY':
        masked_array = masked_tensor.numpy()
        return masked_array
    else:
        print("wrong mode error")
        
def get_negative_mask_tensor(depth_tensor: torch.Tensor, t: int, mode='tensor'):
    """
    一个用于获取背景深度张量. 深度预测结果中, 与数值t相同的部分将被设置为黑色. 即主体部分被设置为白色, 以获取深度张量中的背景部分

    depth_tensor参数：需要可视化的深度预测张量(Tensor), 可以在cpu上, 也可以在gpu上;

    t参数: 需要掩膜的数值, 例如, 如果t=4, 则将深度检测张量中数值不为4的分量全部改成1, 其余部分修改为0
    
    mode参数：用于控制返回值类型, 可以为numpy或tensor, 默认为tensor

    返回：一个掩膜数组, 返回值类型与str参数保持一致
    """
    depth_tensor = depth_tensor.cpu()
    # 将数值不为4的分量全部改成0
    masked_tensor = torch.where(depth_tensor == t, torch.tensor(0, dtype=torch.uint8), torch.tensor(1, dtype=torch.uint8))

    mode = mode.upper()

    if mode == 'TENSOR':
        return masked_tensor
    elif mode == 'NUMPY':
        masked_array = masked_tensor.numpy()
        return masked_array
    else:
        print("wrong mode error")
        
        
def get_range_mask_tensor(depth_tensor: torch.Tensor, list_t: List, mode='tensor'):
    """
    一个用于获取带有掩膜的深度检测张量(Tensor)的函数. 与t相同的数值将被设置为1, 否则被设置为0

    depth_tensor参数：需要可视化的深度预测张量(Tensor), 可以在cpu上, 也可以在gpu上;

    list_t参数: 一个列表, 表示需要掩膜的数值, 例如, 如果t=[1,2,3,4], 则将深度检测张量中数值不为[1,2,3,4]的分量全部改成0, 其余部分修改为1

    mode参数：用于控制返回值类型, 可以为numpy或tensor, 默认为tensor

    返回：一个掩膜数组, 返回值类型与str参数保持一致
    """
    depth_tensor = depth_tensor.cpu()
    
    tensor_t = torch.tensor(list_t)
    # 将数值不为4的分量全部改成0
    masked_tensor = torch.where(torch.isin(depth_tensor, tensor_t), torch.tensor(1, dtype=torch.uint8), torch.tensor(0, dtype=torch.uint8))

    mode = mode.upper()

    if mode == 'TENSOR':
        return masked_tensor
    elif mode == 'NUMPY':
        masked_array = masked_tensor.numpy()
        return masked_array
    else:
        print("wrong mode error")

def get_range_negative_mask_tensor(depth_tensor: torch.Tensor, list_t: List, mode='tensor'):
    """
    一个用于获取背景深度张量. 深度预测结果中, 与数值t相同的部分将被设置为黑色. 即主体部分被设置为白色, 以获取深度张量中的背景部分

    depth_tensor参数：需要可视化的深度预测张量(Tensor), 可以在cpu上, 也可以在gpu上;

    list_t参数: 一个列表, 表示需要掩膜的数值, 例如, 如果t=[1,2,3,4], 则将深度检测张量中数值不为[1,2,3,4]的分量全部改成1, 其余部分修改为0
    
    mode参数：用于控制返回值类型, 可以为numpy或tensor, 默认为tensor

    返回：一个掩膜数组, 返回值类型与str参数保持一致
    """
    depth_tensor = depth_tensor.cpu()
    
    tensor_t = torch.tensor(list_t)
    # 将数值不为4的分量全部改成0
    masked_tensor = torch.where(torch.isin(depth_tensor, tensor_t), torch.tensor(0, dtype=torch.uint8), torch.tensor(1, dtype=torch.uint8))

    mode = mode.upper()

    if mode == 'TENSOR':
        return masked_tensor
    elif mode == 'NUMPY':
        masked_array = masked_tensor.numpy()
        return masked_array
    else:
        print("wrong mode error")

def show_save_mask(mask_tensor: torch.Tensor, save_path_name=None, isshow=True):
    """
    一个用于可视化带有掩膜的深度检测张量(Tensor)的函数.

    depth_tensor参数：需要可视化的深度预测张量(Tensor), 可以在cpu上, 也可以在gpu上;

    t参数: 需要掩膜的数值, 例如, 如果t=4, 则将深度检测张量中数值不为4的分量全部改成0;

    save_path_name参数：保存掩膜图像的位置, 需要提供文件名
    
    isshow参数：控制该函数是否将掩膜可视化, True为显示掩膜, 默认为True

    返回：无返回值.
    """
    if save_path_name is not None:
        mask_tensor.save(save_path_name)
    
    mask_tensor = mask_tensor.cpu()
    
    # 将 Tensor 转换为 NumPy 数组并转换为uint8类型
    masked_array = mask_tensor.numpy().astype(np.uint8)
    print(masked_array)

    masked_image = Image.fromarray(masked_array)  # 'L' 模式表示灰度图像
    
    if isshow:
        # 使用 PIL.Image 显示图像
        masked_image.show()

if __name__ == "__main__":
    import use_metric3d
    import engine
    
    img_path = '/mnt/sda/zxt/z_datas/imgs/1_origin_data/large_perspective_1.png'
    pred_depth = use_metric3d.use_metric3d(rgb_file_path=img_path)
    merged_tensor = engine.merge_depth(pred_depth=pred_depth, t=0)
    print(f"merged_tensor is {merged_tensor}")
    negative_mask = get_mask_tensor(depth_tensor=merged_tensor, t=4, mode='tensor')
    engine.display_save_depth(negative_mask)