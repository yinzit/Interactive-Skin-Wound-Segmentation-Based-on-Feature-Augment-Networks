class DefaultConfig(object):
    num_epochs = 100
    epoch_start_i = 0
    checkpoint_step = 5  # 无用
    validation_step = 1  # 每训练几个epoch进行一次验证
    crop_height = 512
    crop_width = 512
    batch_size = 1  ########################
    input_channel = 3  # 输入的图像通道
    
    data = r'E:\JBHI\JBHI'  # 数据存放的根目录
    wound = "Foot Segmentation"
    log_dirs = r'E:\JBHI\JBHI\wound'  # 存放 tensorboard log的文件夹()

    lr = 0.01
    lr_mode = 'poly'  # poly优化策略
    net_work = 'CE_Net'
    # 可选网络：UNet CE_Net ResNet34  AttU_Net  CPFNet  DANet  PSPNet CSNet FANet MobileNetV2
    momentum = 0.9  # 优化器动量
    weight_decay = 1e-4  # L2正则化系数

    mode = 'test'
    # k_fold = 2
    # test_fold = 3
    model = 'model_059_0.8981.pth.tar'
    num_workers = 8
    num_classes = 1  # 分割类别数，二类分割设置为1，多类分割设置成 类别数+加背景
    cuda = '0'
    use_gpu = True
    # pretrained_wound_model_path = '/home/jason/PycharmProjects/Pytorch_Wound_Segmention/Wound_Code/UNet/checkpoints/' \
    #                               + f'{test_fold}' + '/' + model
    # pretrained_wound_model_path = r'E:\JBHI\JBHI\xiaorong_2/'+ 'EFA/1/' + model
    pretrained_wound_model_path = r'E:\JBHI\JBHI\checkpoint_1\cedi\1\model_052_0.8681.pth.tar'
    save_model_path = './checkpoint_1/cedi'
