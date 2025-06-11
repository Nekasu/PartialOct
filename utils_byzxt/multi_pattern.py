from PIL import Image

def extract_subject(image_path, threshold=200):
    # 打开图像并转换为RGBA模式以处理透明度
    image = Image.open(image_path).convert('RGBA')
    data = image.getdata()
    
    new_data = []
    for item in data:
        # 设定阈值，提取主体，去除白色背景
        if item[0] > threshold and item[1] > threshold and item[2] > threshold:
            new_data.append((255, 255, 255, 0))  # 设为透明
        else:
            new_data.append(item)
    
    image.putdata(new_data)
    return image

def create_pattern(image, canvas_size, overlap=0.3, istransparent=True):
    # 创建一个空白的画布
    if istransparent:
        # 背景透明 
        canvas = Image.new('RGBA', canvas_size, (255, 255, 255, 0))
    else:
        # 背景为白色
        canvas = Image.new('RGBA', canvas_size, (255, 255, 255, 255))
        
    
    img_w, img_h = image.size
    step_x = int(img_w * (1 - overlap))
    step_y = int(img_h * (1 - overlap))
    
    for x in range(-1*step_x//2, canvas_size[0], step_x):
        for y in range(-1*step_y//2, canvas_size[1], step_y):
            # 交错排列
            offset_x = step_x // 2 if (y // step_y) % 2 == 1 else 0
            offset_y = step_y // 2 if (x // step_x) % 2 == 1 else 0
            canvas.paste(image, (x - offset_x, y - offset_y), image)
    
    return canvas

def main(input_image_path, output_image_path, canvas_size=(800, 800), overlap=0.3, istransparent=True):
    # 提取不规则主体
    subject_image = extract_subject(input_image_path)
    # 创建并保存覆盖整个图像的图案
    pattern_image = create_pattern(subject_image, canvas_size, overlap, istransparent=istransparent)
    pattern_image.save(output_image_path)
    print(f'save Successfully at {output_image_path}')


if __name__=='__main__':
    # 示例调用
    input_path_name = '/mnt/sda/zxt/4_temp_dataset/dunhuang_temp/2_part_style_image/background/style_image/dunhuang_back.jpg'
    output_path_name = '/mnt/sda/zxt/4_temp_dataset/dunhuang_temp/2_part_style_image/background/style_image/multi_pattern_dunhuang_back.png'
    main(input_image_path=input_path_name, output_image_path=output_path_name, canvas_size=(800, 800), overlap=0.7, istransparent=True)
