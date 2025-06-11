'''
将掩膜图像里的黑白互换, 原本黑色的部分变成白色、白色的地方变成黑色
'''
from get_images import get_data
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from os import path
def mask_inverse(img:Image):
    tsr_img: torch.Tensor = ToTensor()(img)
    # print(tsr_img)
    tsr_img_ivs = torch.where(tsr_img <= 0.3, torch.tensor(1.0), torch.tensor(0.0))
    tsr_img_ivs = (tsr_img_ivs * 255).to(torch.uint8)
    tsr_img_ivs = tsr_img_ivs.permute(1,2,0)
    pil_img_ivs = Image.fromarray(tsr_img_ivs.numpy())
    # print(tsr_img_ivs)
    return pil_img_ivs
    # img_list = get_data(path)
    # print(img_list)
    # for i, img_p in enumerate(img_list):
    #     img = Image.open(img_p)

def main():
    origin_dir = '/mnt/sda/zxt/3_code_area/code_develop/PartialConv_AesFA/imgs/mask/background'
    save_dir = '/mnt/sda/zxt/3_code_area/code_develop/PartialConv_AesFA/imgs/mask/main'
    t = 1
    for i, img_path in enumerate(get_data(origin_dir)):
        img_name = img_path.split('/')[-1]
        save_path = path.join(save_dir, img_name)

        print(f'origin mask in {img_path}, save to {save_path}')
        t += 1

        img = Image.open(img_path)
        new_img  = mask_inverse(img)
        new_img.save(save_path)
        
    print('compelet')
if __name__ == '__main__':
    main()