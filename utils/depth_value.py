"""
此程序用于探索深度信息：属于同一类别的物体在深度估计的数值方面有什么类似之处.
探索方法如下：以图像/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/000000000001_merge.jpg为基础, 查看/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/enhanced_000000000001.jpg中属于同一类别的物体的深度信息, 打印在屏幕上.
"""

from . import engine
from . import use_metric3d
from . import mask_tensor
##----------------------------------model use-------------------------------------##
img_file_pth = '/mnt/sda/zxt/z_datas/imgs/1_origin_data/large_perspective_1.png'
masked_path = '/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/masked.png'
pred_depth = use_metric3d.use_metric3d(rgb_file_path=img_file_pth)
##--------------------------------------------------------------------------------##

#### you can now do anything with the metric depth 

merged = engine.merge_depth(pred_depth=pred_depth, t=0)

## 保留距离为4m的图像, 距离不为4的像素设置为 0
mask_tensor.show_save_mask_tensor(depth_tensor=merged, t=4, save_path=masked_path)

# for i in range(400, 480):
#     for j in range(300,400):
#         print(f"at {i,j}, the value is {pred_depth[i,j]-3}")
        
# 当前正在尝试：保留到整数部分进行分隔, 将整数部分相同的作为同一图层(即差别在1m以内的物体当做同一个主体)
# 但这么做可能引发一个问题, 即具有大透视的物体难以取得很好的效果, 下面进行测试