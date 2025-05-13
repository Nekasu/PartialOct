from posixpath import basename
from PIL import Image
import os
# from Config import Config

def add_alpha_channel(images_dir, masks_dir, output_dir):
    """
    将掩膜作为alpha通道添加到对应图像，并保存为PNG格式
    
    参数：
        images_dir: 原图文件夹路径
        masks_dir: 掩膜文件夹路径
        output_dir: 输出文件夹路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历原图文件夹
    for img_name in os.listdir(images_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg')):
            # 构造文件路径
            img_path = os.path.join(images_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            
            mask_name = 'style' + base_name + '_mask' + '.jpg'
            # base_name_list = base_name.split('_') 
            # mask_name = base_name_list[0] + '_' + 'mask' + '_' + '_'.join(base_name_list[2:]) + '.jpg'
            print(mask_name)
            # mask_name = f"{base_name}_mask.jpg"
            mask_path = os.path.join(masks_dir, mask_name)

            if not os.path.exists(mask_path):
                print(f"警告：未找到 {mask_name}，已跳过 {img_name}")
                continue

            try:
                # 打开图像并添加Alpha通道
                img = Image.open(img_path).convert("RGBA")
                mask = Image.open(mask_path).convert("L")  # 转换为灰度

                # 确保尺寸一致
                if img.size != mask.size:
                    mask = mask.resize(img.size, Image.Resampling.LANCZOS)

                # 添加Alpha通道
                img.putalpha(mask)

                # 保存为PNG
                output_path = os.path.join(output_dir, f"{base_name}.png")
                img.save(output_path, "PNG")
                print(f"已处理：{img_name} -> {os.path.basename(output_path)}")

            except Exception as e:
                print(f"处理 {img_name} 时出错：{str(e)}")

if __name__ == "__main__":
    # 配置路径（根据实际情况修改）
    IMAGES_DIR = '/mnt/sda/Datasets/style_image/dunhuang_style/enhanced/main_white/origin'
    MASKS_DIR = '/mnt/sda/Datasets/style_image/dunhuang_style/enhanced/main_white/mask'
    # IMAGES_DIR = Config.style_dir
    # MASKS_DIR = Config.mask_dir
    OUTPUT_DIR = '/mnt/sda/Datasets/style_image/dunhuang_style/enhanced/main_white/alpha'

    add_alpha_channel(IMAGES_DIR, MASKS_DIR, OUTPUT_DIR)