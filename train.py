import os
import torch
import numpy as np
import tensorboardX

from Config import Config
from DataSplit import DataSplit
from model import AesFA
from blocks import model_save, model_load, update_learning_rate

from torch.utils.data import RandomSampler # Load Content Images

def mkoutput_dir(config):
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)

def get_n_params(model):
    total_params=0
    net_params = {'netE':0, 'netS':0, 'netG':0, 'vgg_loss':0}

    for name, param in model.named_parameters():
        net = name.split('.')[0]
        nn=1
        for s in list(param.size()):
            nn = nn*s
        net_params[net] += nn
        total_params += nn
    return total_params, net_params

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach().numpy()
    image = image.transpose(0, 2, 3, 1)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

def main():
    config = Config()
    mkoutput_dir(config)

    config.device = torch.device(config.cuda_device if torch.cuda.is_available() else "cpu")
    print('cuda:', config.device)

    print('Version:', config.file_n)
    
    ########## Data Loader ##########
    train_data = DataSplit(config=config, phase='train')
    train_sampler = RandomSampler(train_data)
    data_loader_train = torch.utils.data.DataLoader(
        train_data,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=False,
        sampler=train_sampler
    )
    with open('./Config.py', 'r') as f:
        print(f.read())
    print("Train: ", train_data.__len__(), "images: ", len(data_loader_train), "x", config.batch_size,"(batch size) =", train_data.__len__())

    ########## load model ##########
    model = AesFA(config)
    model.to(config.device)
    
    # # of parameter
    param_num, net_params = get_n_params(model)
    print("# of parameter:", param_num)
    print("parameters of networks:", net_params)

    ########## load saved model - to continue previous learning ##########
    if config.train_continue == 'on':
        model, model.optimizer_E, model.optimizer_S, model.optimizer_G, epoch_start, tot_itr = model_load(checkpoint=None, ckpt_dir=config.ckpt_dir, model=model,
                           optim_E=model.optimizer_E,
                           optim_S=model.optimizer_S,
                           optim_G=model.optimizer_G)
        print(epoch_start, "th epoch ", tot_itr, "th iteration model load")
    else:
        epoch_start = 0
        tot_itr = 0

    # model = model.double()
    train_writer = tensorboardX.SummaryWriter(config.log_dir)

    ########## Training ##########
    # to save ckpt file starting with epoch and iteration 1
    epoch = epoch_start - 1
    tot_itr = tot_itr - 1
    while tot_itr < config.n_iter:
        epoch += 1
        for i, data in enumerate(data_loader_train):
            # for k,v in data.items():
            #     data[k] = data[k].double()
            #     print(data[k].max(), data[k].min())
                # v_m = v.mean()
                # v_std = v.std()
                # data[k] = (v - v_m) / (v_std + 1e-7)
            # print(f"+++++++++++++++++++++++++++++在第 {epoch} 个epoch中的第 {i} 个循环中++++++++++++++++++++++++++++++++")
            # data 是一个存放数据的字典, 其中具有三个键值对, 三个键分别为 content_img, style_img, mask_img

            # print(f'cotent_image is nan test: {torch.isnan(data["content_img"]).any()}')
            # print(f'cotent_image is inf test: {torch.isinf(data["content_img"]).any()}')

            # print(f'style_image is nan test: {torch.isnan(data["style_img"]).any()}')
            # print(f'style_image is inf test: {torch.isinf(data["style_img"]).any()}')

            # print(f'mask_image is nan test: {torch.isnan(data["mask_img"]).any()}')
            # print(f'mask_image is inf test: {torch.isinf(data["mask_img"]).any()}')
            # print(f'掩膜图像：{data["mask_img"]}')
            tot_itr += 1
            # breakpoint()
            train_dict = model.train_step(data) # 核心训练代码
            # breakpoint()

            real_A = im_convert(data['content_img'])
            real_B = im_convert(train_dict['style_img'])
            fake_B = im_convert(train_dict['fake_AtoB'])
            trs_high = im_convert(train_dict['fake_AtoB_high'])
            trs_low = im_convert(train_dict['fake_AtoB_low'])
            # ------------------------------问题在这之间 ------------------------------
            ## Tensorboard ##
            # tensorboard - loss
            train_writer.add_scalar('Loss_G', train_dict['G_loss'], tot_itr)
            # print(f"train_dict['G_loss'] is nan test: {torch.isnan(train_dict['G_loss']).any()}")
            # print(f"train_dict['G_loss'] is inf test: {torch.isinf(train_dict['G_loss']).any()}")

            train_writer.add_scalar('Loss_G_Percept', train_dict['G_Percept'], tot_itr)
            # print(f"train_dict['G_Percept'] is nan test: {torch.isnan(train_dict['G_Percept']).any()}")
            # print(f"train_dict['G_Percept'] is inf test: {torch.isinf(train_dict['G_Percept']).any()}")
            
            train_writer.add_scalar('Loss_G_Contrast', train_dict['G_Contrast'], tot_itr)
            # print(f"train_dict['G_Contrast'] is nan test: {torch.isnan(train_dict['G_Contrast']).any()}")
            # print(f"train_dict['G_Contrast'] is inf test: {torch.isinf(train_dict['G_Contrast']).any()}")
            # breakpoint()
            
            # tensorboard - images
            train_writer.add_image('Content_Image_A', real_A, tot_itr, dataformats='NHWC')
            train_writer.add_image('Style_Image_B', real_B, tot_itr, dataformats='NHWC')
            train_writer.add_image('Generated_Image_AtoB', fake_B, tot_itr, dataformats='NHWC')
            train_writer.add_image('Translation_AtoB_high', trs_high, tot_itr, dataformats='NHWC')
            train_writer.add_image('Translation_AtoB_low', trs_low, tot_itr, dataformats='NHWC')
            # ------------------------------问题在这之间 ------------------------------

            print("Tot_itrs: %d/%d | Epoch: %d | itr: %d/%d | Loss_G: %.5f"%(tot_itr+1, config.n_iter, epoch+1, (i+1), len(data_loader_train), train_dict['G_loss']))

            if (tot_itr + 1) % 10000 == 0:
                model_save(ckpt_dir=config.ckpt_dir, model=model, optim_E=model.optimizer_E, optim_S=model.optimizer_S, optim_G=model.optimizer_G, epoch=epoch, itr=tot_itr)
                print(tot_itr+1, "th iteration model save")

            # print(f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        update_learning_rate(model.E_scheduler, model.optimizer_E)
        update_learning_rate(model.S_scheduler, model.optimizer_S)
        update_learning_rate(model.G_scheduler, model.optimizer_G)

if __name__ == '__main__':
    config = Config()
    print(config.log_file_path)
    main()
