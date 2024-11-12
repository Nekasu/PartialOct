from torchvision.transforms import RandomCrop
from path import Path
from PIL import Image

def main():
    dataset_path = '/mnt/sda/Dataset/style_image/dunhuang_style'
    input_image_path = '/base/origin'
    input_mask_path = '/base/mask'
    output_image_path = '/crop_256/origin'
    output_mask_path = '/crop_256/mask'
    crop_size_tuple = (256,256)
    rc = RandomCrop(size=crop_size_tuple)

    input_image_path = Path(dataset_path + input_image_path)
    input_mask_path = Path(dataset_path + input_mask_path)
    
    output_image_path = Path(dataset_path + output_image_path)
    output_mask_path = Path(dataset_path + output_mask_path)

    input_image_list = get_data(input_image_path)
    input_mask_list = get_data(input_mask_path)
    
    img_and_mask = list(zip(input_image_list, input_mask_list))
    l = []
    for index, path in enumerate(img_and_mask):
        try:
            print(f'正在处理第{index+1}个图像及其掩膜')
            print(path[0].split('/')[-1], path[1].split('/')[-1])
            # 多次随机裁剪以扩充数据集，例如每对图像和掩膜生成5个裁剪样本
            for crop_index in range(5):
                cropped_image, cropped_mask = do_crop(random_cropper=rc, image_path=path[0], mask_path=path[1])
            
                # 构建唯一的文件名
                cropped_image_path = output_image_path / f"cropped_image_{index}_{crop_index}.png"
                cropped_mask_path = output_mask_path / f"cropped_mask_{index}_{crop_index}.png"
            
                # 保存裁剪后的图像和掩膜
                cropped_image.save(cropped_image_path)
                cropped_mask.save(cropped_mask_path)
            
                print(f'处理完成, 剪切图像位于{cropped_image_path}与{cropped_mask_path}')
        except:
            l.append(index+1)
    for i, j in enumerate(l):
        print(f"处理第 {j}个图像时出现错误")        
    
    
    

def get_data(img_dir):
    file_type = ['*.jpg', '*.png', '*.jpeg', '*.tif']
    imgs = []
    for ft in file_type:
        imgs += sorted(img_dir.glob(ft))
    images = sorted(imgs)
    return images

def do_crop(random_cropper, image_path, mask_path):
    # 读入风格图像
    origin_image = Image.open(image_path).convert('RGB')
    # 读入掩膜图像
    mask_image = Image.open(mask_path).convert('RGB')
    
    combined = Image.merge('RGBA', (origin_image.split()[0], origin_image.split()[1], origin_image.split()[2], mask_image.split()[0]))
    
    cropped_combined = random_cropper(combined)
    cropped_image  = cropped_combined.convert('RGB')
    cropped_mask  = cropped_combined.split()[-1].convert('RGB')
    
    return cropped_image, cropped_mask
    

if __name__ == '__main__':
    main()