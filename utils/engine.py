
import cv2
import torch
import numpy as np
from PIL import Image

def write_depth(pred_depth, file_path_name)   -> None:
    """
    一个用于将深度信息保存起来的函数.
    以张量(Tensor)的形式传入深度信息，将其保存到指定路径的文件中.
    
    pred_depth参数：存储深度信息的张量(Tensor).
    
    file_path_name参数：保存深度信息的文件路径, 需要给出文件名
    
    返回：无返回值.
    """
    with open(file=file_path_name, mode='w') as f:
        for i in range(pred_depth.size(0)):
            f.write(str(pred_depth[i])+'\n')
    
    print(f"成功写入文件{file_path_name}！")

#### ajust input size to fit pretrained model
def adjust_input_size(image, intrinsic):# -> tuple[Any, list, list]:
    """
    一个用于调整输入图像大小以适应预训练模型的函数.
    image参数：输入的图像.
    intrinsic参数：相机内参.
    返回：处理后的图像, pad_info, 调整后的相机内参.
    """
    # keep ratio resize
    input_size = (616, 1064) # for vit model
    # input_size = (544, 1216) # for convnext model
    h, w = image.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    # remember to scale intrinsic, hold depth
    intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
    # padding to input_size
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    return rgb, pad_info, intrinsic

def normalize(image):
    """
    一个用于将图像归一化的函数.
    image参数：输入的图像.
    返回：归一化后的图像.
    """
    #### normalize
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    image = torch.from_numpy(image.transpose((2, 0, 1))).float()
    image = torch.div((image - mean), std)
    image = image[None, :, :, :].cuda()
    return image

def unpad(pred_depth, rgb_origin, pad_info) -> torch.Tensor:
    """
    一个用于去除padding的函数.
    pred_depth参数：预测的深度图.
    rgb_origin参数：原始图像, 用于获取原始大小, 从而为去除padding后的深度图上采样.
    pad_info参数：padding信息.
    """
    # un pad
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]

    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()
    return  pred_depth

def decanonical_transform(intrinsic, pred_depth):
    """
    一个用于将预测的深度图从规范化空间转换为实际空间的函数.

    即论文中的de-canonical transform.

    intrinsic参数：相机内参.

    pred_depth参数：预测的深度图.
    
    返回：转换后的深度图.
    """
        #### de-canonical transform
    canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
    pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
    pred_depth = torch.clamp(pred_depth, 0, 300)
    return pred_depth

# 整合预测的深度信息, 仅保留到小数点后t位 
def merge_depth(pred_depth, t)->torch.Tensor:
    """
    一个用于整合深度预测结果的函数, 深度预测的结果将保留到小数点后t位.

    pred_depth变量：需要整合深度的张量(Tensor), 传入在cpu中的Tensor与在gpu中的Tensor均可.

    t变量：每个张量保留到小数点后多少位.

    返回：保留到小数点后t位的张量(Tensor)
    """
    temp = pred_depth.cpu().numpy()
    merged = temp
    for i in range(pred_depth.size(0)):
        for j in range(pred_depth.size(1)):
            merged[i][j] = round(temp[i][j], t)
    
    merged = torch.from_numpy(merged)
    return merged

def get_enhanced_depth(pred_depth) -> np.ndarray:
    """
    本函数作为一个工具函数, 不应该在主函数中调用.

    本函数用于在可视化深度预测结果时, 防止深度预测结果张量(Tensor)中的差别过小而进行的图像增强.

    pred_depth参数：预测的深度图, 传入在cpu中的Tensor与在gpu中的Tensor均可.

    返回：增强后的深度图, 供其他函数使用, 是一个numpy.ndarray
    """
     # 如果张量是三维的，需要去掉第一个维度
    if pred_depth.dim() == 3:
        pred_depth = pred_depth.squeeze(0)

    # 将张量转换为NumPy数组
    pred_depth.cpu().numpy()
    depth_array = pred_depth.cpu().numpy()

    # 归一化到 0-255 之间
    depth_array = (255 * (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())).astype(np.uint8)

    # 应用CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    depth_array_clahe = clahe.apply(depth_array)
    return depth_array_clahe
    
def display_save_depth(pred_depth: torch.Tensor, path_name=None) -> None:
    """
    一个用于将深度预测结果可视化的函数.

    pred_depth参数：预测的深度图, 传入在cpu中的Tensor与在gpu中的Tensor均可.

    path_name参数：保存深度图可视化结果的路径, 需要给出文件名

    返回：无返回值.
    """
    depth_array_clahe = get_enhanced_depth(pred_depth=pred_depth)
    # 使用Matplotlib显示图像
    depth_array_clahe_image = Image.fromarray(depth_array_clahe)
    depth_array_clahe_image.show()
    
    if path_name != None:
        depth_array_clahe_image.save(fp=path_name)
    
def main() -> None:
    #### prepare data
        # 图像路径
    rgb_file = '/mnt/sda/zxt/z_datas/imgs/3_style_data/fu_rong.jpg'
        # 将可视化深度信息图像保存的路径
    origin_rgb_file_name = rgb_file.split(sep='/')[-1]
    visible_file_path = '/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/enhanced_'+origin_rgb_file_name
        # 相机内参
    intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]
    gt_depth_scale = 256.0
    
        # 读取图像
    rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]
        # 调整输入图像大小以适应预训练模型
    rgb, pad_info, intrinsic = adjust_input_size(image=rgb_origin, intrinsic=intrinsic)
        # 归一化
    rgb = normalize(rgb)

    # model loading
    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    model.cuda().eval()

    # model inference
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({'input': rgb})
        
    
    # post processing
    pred_depth = unpad(pred_depth, rgb_origin, pad_info)
    pred_depth = decanonical_transform(intrinsic=intrinsic, pred_depth=pred_depth)
    ##--------------------------------------------------------------------------------##
    #### you can now do anything with the metric depth 
    # such as print it on the screen
    # print(pred_depth.shape)
    # print(pred_depth)

    # # or save it to a file
    # pred_depth= pred_depth.cpu()
    # write_depth(file_path=file_path, pred_depth=pred_depth)

    # # or display and save it
    
    # merged = merge_depth(pred_depth=pred_depth, t=0)
    # print(f"after merging, the depth_info is {merged}")
    display_save_depth(pred_depth=pred_depth, path=visible_file_path)
    
if __name__ == '__main__':
    main()