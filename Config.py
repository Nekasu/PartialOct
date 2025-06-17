class Config:
    # phase = 'train'         # You must change the phase into train/test/style_blending
    phase = 'test'         # You must change the phase into train/test/style_blending
    train_continue = 'off'  # on / off

    data_num = 60000        # Maximum # of training data

    content_dir = '/mnt/sda/Datasets/Detection/COCO/train2017'
    # style_dir = '/mnt/sda/Dataset/style_image/dunhuang_style/crop_256/main_white/origin'
    # mask_dir = '/mnt/sda/Dataset/style_image/dunhuang_style/crop_256/main_white/mask'
    style_dir = '/mnt/sda/Datasets/style_image/AlphaStyle/alpha_WikiArt_AllInOne2/train_alpha'
    # mask_dir = '/mnt/sda/Dataset/style_image/dunhuang_style/crop_256/main_white/mask'
    cuda_device = 'cuda:1'
    
    file_n = 'SoftPartialConv_limited' # 新训练时, 此处需要修改. 测试时, 也需要修改此处
    log_file_path = '/mnt/sda/zxt/3_code_area/code_develop/PartialConv_AesFA/log/' + file_n + 'log.txt'
    log_dir = './log/' + file_n
    ckpt_dir = './ckpt/' + file_n
    img_dir = './Generated_images/' + file_n
    
    if phase == 'test':
        multi_to_multi = True
        test_content_size = 256
        test_style_size = 256

        mod = 'main' # main 表示迁移风格图像, back表示迁移背景图像, 当该值为main时, 将使用主体图像进行风格迁移；当该值为 back 时, 将使用背景图像进行风格迁移. 同时, 该值还用于创建保存图像的文件名.
        content_dir = '/mnt/sdb/zxt/3_code_area/code_develop/PartialConv_AesFA/input/contents/alpha'
        # style_dir = '/mnt/sdb/zxt/3_code_area/code_develop/PartialConv_AesFA/input/styles/dunhuang/alpha'  # 敦煌
        style_dir = '/mnt/sdb/zxt/3_code_area/code_develop/PartialConv_AesFA/input/styles/wikiart/alpha'    # wikiart
        style_dir = '/mnt/sdb/zxt/3_code_area/code_develop/PartialConv_AesFA/input/styles/wikiart/alpha'    # wikiart


        # mask_dir = '/mnt/sda/zxt/3_code_area/code_develop/PartialConv_AesFA/imgs/masks' + '/' + mod
        # style_dir = '/mnt/sda/zxt/3_code_area/code_develop/PartialConv_AesFA/imgs/styles/origin'
        # mask_dir = '/mnt/sda/zxt/3_code_area/code_develop/PartialConv_AesFA/imgs/styles/mask'

        ckpt_iter = 160000
        ckpt_epoch = 22
        ckpt_name = 'model_iter_' + str(ckpt_iter) + '_epoch_' + str(ckpt_epoch) + '.pth'
         
        img_dir = './output/'+file_n + '/' + ckpt_name.split('.')[0] + '_' + str(test_content_size)+  '/'

    elif phase == 'style_blending':
        blend_load_size = 256
        blend_dir = './blendingDataset/'
        content_img = blend_dir + str(blend_load_size) + '/A.jpg'
        style_high_img = blend_dir + str(blend_load_size) + '/B.jpg'
        style_low_img = blend_dir + str(blend_load_size) + '/C.jpg'
        img_dir = './output/'+file_n+'_blending_' + str(blend_load_size)
        
    # VGG pre-trained model
    vgg_model = './vgg_normalised.pth'

    ## basic parameters
    n_iter = 160000
    batch_size = 8
    lr = 0.00001
    lr_policy = 'step'
    lr_decay_iters = 50
    beta1 = 0.0

    # preprocess parameters
    load_size = 512
    crop_size = 256

    # model parameters
    input_nc = 3         # of input image channel
    nf = 64              # of feature map channel after Encoder first layer
    output_nc = 3        # of output image channel
    style_kernel = 3     # size of style kernel
    
    # Octave Convolution parameters
    alpha_in = 0.5       # input ratio of low-frequency channel
    alpha_out = 0.5      # output ratio of low-frequency channel
    freq_ratio = [1, 1]  # [high, low] ratio at the last layer

    # Loss ratio
    lambda_percept = 1.0
    lambda_perc_cont = 1.0
    lambda_perc_style = 10.0
    lambda_const_style = 5.0

    # Else
    norm = 'instance'
    init_type = 'normal'
    init_gain = 0.02
    no_dropout = 'store_true'
    num_workers = 4
