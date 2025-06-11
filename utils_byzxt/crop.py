from torchvision.transforms import RandomCrop
from path import Path
from PIL import Image
from get_images import get_data

def main():
    dataset_path = '/mnt/sda/Dataset/style_image/dunhuang_style'

    input_image_path = '/enhanced/main_white/origin'
    input_mask_path = '/enhanced/main_white/mask'

    output_image_path = '/crop_256/main_white/origin'
    output_mask_path = '/crop_256/main_white/mask'
    crop_size_tuple = (256,256)
    rc = RandomCrop(size=crop_size_tuple)

    input_image_path = Path(dataset_path + input_image_path)
    input_mask_path = Path(dataset_path + input_mask_path)
    
    output_image_path = Path(dataset_path + output_image_path)
    output_mask_path = Path(dataset_path + output_mask_path)

    input_image_list = get_data(input_image_path)
    input_mask_list = get_data(input_mask_path)
    print(f"input image list is {input_image_list}")
    print(f"input mask list is {input_mask_list}")
    img_and_mask = list(zip(input_image_list, input_mask_list))
    l = []
    for index, path in enumerate(img_and_mask):
        try:
            print(f'--------------------------正在处理第{index+1}个图像及其掩膜--------------------------')
            print(path[0].split('/')[-1], path[1].split('/')[-1])
            # 多次随机裁剪以扩充数据集，例如每对图像和掩膜生成5个裁剪样本
            for crop_index in range(5):
                flag = True
                while flag:
                    cropped_image, cropped_mask = do_crop(random_cropper=rc, image_path=path[0], mask_path=path[1])
                
                    if not isblack(cropped_mask):
                        print(f"获取第 {index+1} 张图像的第 {crop_index+1} 份同时具有主体与背景的图像, 正在保存")
                        # 构建唯一的文件名
                        cropped_image_path = output_image_path / f"cropped_image_{index+1}_{crop_index}.jpg"
                        print(cropped_image_path)
                        cropped_mask_path = output_mask_path / f"cropped_mask_{index+1}_{crop_index}.jpg"
                        print(cropped_mask_path)
                        # 保存裁剪后的图像和掩膜
                        cropped_image.save(cropped_image_path)
                        cropped_mask.save(cropped_mask_path)
                
                        print(f'保存完成, 剪切图像位于{cropped_image_path}与{cropped_mask_path}')
                        flag = False
                    else:
                        print(f"第 {index+1} 张图像的第 {crop_index+1} 份裁剪为纯黑, 重新裁剪 ")
                        flag = True
        except:
            l.append(index+1)
    for i, j in enumerate(l):
        print(f"处理第 {j}个图像时出现错误")        

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
    
def isblack(input_image):
    pixels = input_image.load()
    for y in range(input_image.size[1]):
        for x in range(input_image.size[0]):
            r, g, b= pixels[x, y]
            if r!=0 or g!=0 or b!=0:
                return False
    return True
                

if __name__ == '__main__':
    main()