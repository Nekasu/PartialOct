from . import engine
import torch
import cv2

def use_metric3d(rgb_file_path_name:str):
    """
    一个用于直接调用metric3d模型的函数, 集成了图像读取, 相机内参定义, unpad以及逆标准相机变换的步骤.
    
    rgb_file_path_name参数：需要进行深度预测的文件路径, 需要文件名
    
    返回：深度预测的结果, 一个张量(Tensor).
    """
    origin_rgb_file_name = rgb_file_path_name.split(sep='/')[-1]
    visible_file_path = '/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/enhanced_'+origin_rgb_file_name
        # 相机内参
    intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]
    gt_depth_scale = 256.0
        # 读取图像
    rgb_origin = cv2.imread(rgb_file_path_name)[:, :, ::-1]
        # 调整输入图像大小以适应预训练模型
    rgb, pad_info, intrinsic = engine.adjust_input_size(image=rgb_origin, intrinsic=intrinsic)
        # 归一化
    rgb = engine.normalize(rgb)

    # model loading
    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    model.cuda().eval()

    # model inference
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({'input': rgb})
        

    # post processing
    pred_depth = engine.unpad(pred_depth, rgb_origin, pad_info)
    pred_depth = engine.decanonical_transform(intrinsic=intrinsic, pred_depth=pred_depth)
    
    return pred_depth

if __name__ == '__main__':
    rgb_file_path_name = '/mnt/sda/Dataset/Detection/COCO/test2017/000000000001.jpg'
    print(use_metric3d(rgb_file_path_name=rgb_file_path_name))