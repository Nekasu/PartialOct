'''
Use model.py->AesFA_test->forward function to generate stylized images.
'''
import os
from unittest import result
from cv2 import transform
import path
import torch
import numpy as np
import thop
from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize, RandomCrop

from Config import Config
from DataSplit import DataSplit
from model import AesFA_test
from blocks import test_model_load
import generate_results_html


def load_img(img_name, img_size, device):
    img = Image.open(img_name).convert('RGBA') # 加载图像, 并转换为 RGBA 格式
    img = do_base_transform(img, img_size).to(device)  # 进行剪切以及 ToTensor操作, 这个操作对掩膜部分与内容/风格部分都是必要的
    
    # 从整个图像中提取掩膜图像
    mask_img = img[3,:,:]
    mask_img = mask_img.unsqueeze(0)
    mask_img = mask_img.repeat(3,1,1)
    if len(mask_img) == 3:
        mask_img = mask_img.unsqueeze(0)

    # 从整个图像中提取内容/风格图像
    true_img = img[0:3,:,:]
    if len(true_img) == 3:
        true_img = true_img.unsqueeze(0)
    true_img = do_normalize_transform(true_img) # 进行归一化等操作
    # print(f'mask_img.shape: {mask_img.shape}, true_img.shape:{true_img.shape}')
    
    return true_img, mask_img
    

    # img = do_transform(img, img_size).to(device)
    # if len(img.shape) == 3:
    #     img = img.unsqueeze(0)  # make batch dimension
    # return img

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach().numpy()
    image = image.transpose(0, 2, 3, 1)
    # image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image * 0.5 + 0.5    # 修改为与输入归一化参数对应的值
    image = image.clip(0, 1)
    return image

def do_base_transform(img, osize):
    transform = Compose([Resize(size=osize),
                        CenterCrop(size=osize),
                            ToTensor()])
    return transform(img)
    
def do_normalize_transform(img):
    # transform = Compose([Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    transform = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform(img)

def do_transform(img, osize):
    transform = Compose([Resize(size=osize),  # Resize to keep aspect ratio
                        CenterCrop(size=osize),
                        ToTensor(),
                        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return transform(img)

def save_img(config, cont_name, sty_name, content, style, stylized, freq=False, high=None, low=None):
    real_A = im_convert(content)
    real_B = im_convert(style)
    trs_AtoB = im_convert(stylized)
    
    A_image = Image.fromarray((real_A[0] * 255.0).astype(np.uint8))
    B_image = Image.fromarray((real_B[0] * 255.0).astype(np.uint8))
    trs_image = Image.fromarray((trs_AtoB[0] * 255.0).astype(np.uint8))

    A_path = f"{config.img_dir}/{cont_name.stem}_content_{sty_name.stem}.png"
    B_path = f"{config.img_dir}/{cont_name.stem}_style_{sty_name.stem}.png"
    trs_path = f"{config.img_dir}/{cont_name.stem}_stylized_{sty_name.stem}.png"

    # A_path = '{}/{:s}_content_{:s}.jpg'.format(config.img_dir, cont_name.stem, sty_name.stem)
    A_image.save(A_path)
    # B_path = '{}/{:s}_style_{:s}.jpg'.format(config.img_dir, cont_name.stem, sty_name.stem)
    B_image.save(B_path)
    # trs_path ='{}/{:s}_stylized_{:s}.jpg'.format(config.img_dir, cont_name.stem, sty_name.stem) 
    trs_image.save(trs_path)
    
    if freq:
        trs_AtoB_high = im_convert(high)
        trs_AtoB_low = im_convert(low)

        trsh_image = Image.fromarray((trs_AtoB_high[0] * 255.0).astype(np.uint8))
        trsl_image = Image.fromarray((trs_AtoB_low[0] * 255.0).astype(np.uint8))
        
        trsh_image.save('{}/{:s}_stylizing_high_{:s}.jpg'.format(config.img_dir, cont_name.stem, sty_name.stem))
        trsl_image.save('{}/{:s}_stylizing_low_{:s}.jpg'.format(config.img_dir, cont_name.stem, sty_name.stem))
    
    return A_path, B_path, trs_path

        
def main():
    config = Config()

    device = torch.device(config.cuda_device if torch.cuda.is_available() else 'cpu')
    print('Version:', config.file_n)
    print(device)
    
    with torch.no_grad():
        ## Data Loader
        test_bs = 1
        test_data = DataSplit(config=config, phase='test')
        data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=test_bs, shuffle=False, num_workers=16, pin_memory=False)
        print("Test: ", test_data.__len__(), "images: ", len(data_loader_test), "x", test_bs, "(batch size) =", test_data.__len__())

        ## Model load
        ckpt = os.path.join(config.ckpt_dir, config.ckpt_name)    # ckpt files path&name is from Config.py
        print("checkpoint: ", ckpt)
        model = AesFA_test(config)
        model = test_model_load(checkpoint=ckpt, model=model)
        model.to(device)

        if not os.path.exists(config.img_dir):
            os.makedirs(config.img_dir)

        ## Start Testing
        freq = False                # whether save high, low frequency images or not
        count = 0
        t_during = 0

        A_path_list = []
        B_path_list = []
        trs_path_list = []

        contents = test_data.images # Load Content Images 一个列表, 里面存储了内容图像的名称.
        styles = test_data.style_images    # Load Style Images 一个列表, 里面存储了风格图像的名称.
        # masks = test_data.mask_images# Load Mask Images
        if config.multi_to_multi:   # one content image, N style image
            tot_imgs = len(contents) * len(styles)
            for idx in range(len(contents)):
                cont_name = contents[idx]           # path of content image
                content, content_mask = load_img(cont_name, config.test_content_size, device)

                for i in range(len(styles)):
                    sty_name = styles[i]            # path of style image
                    style, mask = load_img(sty_name, config.test_style_size, device) # 想要将掩膜从风格图像中提取出来, 就必须改写 load_img 函数. 具体来说, 应该将 load_img 函数改写成类似于 DataSplit.py -> __getitem__函数 中, 从 sty_img 中分离 mask的形式.

                    # mask_name = masks[i]            # path of mask image
                    # mask = load_img(mask_name, config.test_style_size, device)
                    
                    if freq:
                        stylized, stylized_high, stylized_low, during = model(real_A=content, real_B=style, real_mask=mask, freq=freq) # Use `AesFA_test.forward` to generate styled images
                        A_path, B_path, trs_path = save_img(config, cont_name, sty_name, content, style, stylized, freq, stylized_high, stylized_low)
                        A_path_list.append(A_path)
                        B_path_list.append(B_path)
                        trs_path_list.append(trs_path)
                    else:
                        stylized, during = model(real_A=content, real_B=style, real_mask=mask, freq=freq)  # Use `AesFA_test.forward` to generate styled images
                        A_path, B_path, trs_path = save_img(config, cont_name, sty_name, content, style, stylized)
                        A_path_list.append(A_path)
                        B_path_list.append(B_path)
                        trs_path_list.append(trs_path)

                    count += 1
                    print(count, idx+1, i+1, during)
                    t_during += during
                    flops, params = thop.profile(model, inputs=(content, style, mask, freq))
                    print("GFLOPS: %.4f, Params: %.4f"% (flops/1e9, params/1e6))
                    print("Max GPU memory allocated: %.4f GB" % (torch.cuda.max_memory_allocated(device=config.cuda_device) / 1024. / 1024. / 1024.))

        else:
            tot_imgs = len(contents)
            for idx in range(len(contents)):
                cont_name = contents[idx]
                content = load_img(cont_name, config.test_content_size, device)

                sty_name = styles[idx]
                style = load_img(sty_name, config.test_style_size, device)

                if freq:
                    stylized, stylized_high, stylized_low, during = model(content, style, freq)
                    save_img(config, cont_name, sty_name, content, style, stylized, freq, stylized_high, stylized_low)
                else:
                    stylized, during = model(content, style, freq)
                    A_path, B_path, trs_path = save_img(config, cont_name, sty_name, content, style, stylized)
                    A_path_list.append(A_path)
                    B_path_list.append(B_path)
                    trs_path_list.append(trs_path)

                t_during += during
                flops, params = thop.profile(model, inputs=(content, style, freq))
                print("GFLOPS: %.4f, Params: %.4f" % (flops / 1e9, params / 1e6))
                print("Max GPU memory allocated: %.4f GB" % (torch.cuda.max_memory_allocated(device=config.gpu) / 1024. / 1024. / 1024.))


        t_during = float(t_during / (len(contents) * len(styles)))
        print("[AesFA] Content size:", config.test_content_size, "Style size:", config.test_style_size,
              " Total images:", tot_imgs, "Avg Testing time:", t_during)
        generate_results_html.generate_html(A_path_list, B_path_list, trs_path_list, file_type=config.mod)
            
if __name__ == '__main__':
    main()
