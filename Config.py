class Config:
    phase = 'train'         # You must change the phase into train/test/style_blending
    train_continue = 'off'  # on / off

    data_num = 60000        # Maximum # of training data

    content_dir = '/mnt/sda/Dataset/Detection/COCO/train2017'
    style_dir = '/mnt/sda/Dataset/style_image/dunhuang_style/crop_256/main_white/origin'
    mask_dir = '/mnt/sda/Dataset/style_image/dunhuang_style/crop_256/main_white/mask'
    
    file_n = 're-dh-white-main' # 新训练时, 此处需要修改.
    log_file_path = '/mnt/sda/zxt/3_code_area/code_develop/PartialConv_AesFA/log/' + file_n + 'log.txt'
    log_dir = './log/' + file_n
    ckpt_dir = './ckpt/' + file_n
    img_dir = './Generated_images/' + file_n
    
    if phase == 'test':
        multi_to_multi = True
        test_content_size = 256
        test_style_size = 256

        mod = 'back' # main 表示迁移风格图像, back表示迁移背景图像, 该值主要用于创建保存图像的文件名, 对实际功能没有影响
        content_dir = '/mnt/sda/zxt/3_code_area/code_develop/PartialConv_AesFA/imgs/contents/full'
        style_dir = '/mnt/sda/zxt/3_code_area/code_develop/PartialConv_AesFA/imgs/styles'
        mask_dir = '/mnt/sda/zxt/3_code_area/code_develop/PartialConv_AesFA/imgs/masks' + '/' + mod
        # style_dir = '/mnt/sda/zxt/3_code_area/code_develop/PartialConv_AesFA/imgs/styles/origin'
        # mask_dir = '/mnt/sda/zxt/3_code_area/code_develop/PartialConv_AesFA/imgs/styles/mask'

        img_dir = './output/'+file_n + '/' + str(test_content_size) + '/' + mod 

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
    n_iter = 160000 * 5
    batch_size = 8
    lr = 0.0001
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
