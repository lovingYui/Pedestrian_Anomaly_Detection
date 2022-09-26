import os
import sys
import torch

from torch.utils.data import DataLoader

from dataset import SequenceDataset
from loss import GradinetLoss, IntensityLoss,GeneratorAdversarialLoss,DiscriminatorAdversarialLoss
from unet import UNet
from discriminator import PixelDiscriminator
from tqdm import tqdm

import cv2

from time import time

class args():
    epochs = 60   # "number of training epochs, default is 10"
    save_per_epoch = 15
    batch_size = 4  # "batch size for training/testing, default is 1"
    pretrained = False
    save_model_dir = "./weights/" #"path to folder where trained model with checkpoints will be saved."
    save_logs_dir = "./logs/"
    num_workers = 0
    resume = False

    # generator setting
    g_lr_init = 0.0002
    
    # discriminator setting
    d_lr_init = 0.00002

    flownnet_pretrained = 'pretrained/FlowNet2-SD.pth'

    # Dataset setting
    channels = 3
    size = 256
    videos_dir = 'D:\\Study\\SRTP\\UCSD_Anomaly_Dataset.v1p2\\UCSDped2\\Train'
    time_steps = 5
    
    # For GPU training
    gpu = 0

def check_cuda():
    if torch.cuda.is_available() and args.gpu is not None:
        return True
    else:
        return False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train():

    use_cuda = check_cuda()
    print(use_cuda)

    generator = UNet(in_channels=args.channels * (args.time_steps - 1), out_channels=args.channels)
    discriminator = PixelDiscriminator(input_nc=args.channels)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.g_lr_init)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr_init)
    
    intensity_loss = IntensityLoss()
    gradient_loss = GradinetLoss(args.channels)
    adversarial_loss = GeneratorAdversarialLoss()
    discriminator_loss = DiscriminatorAdversarialLoss()
    #搭建生成器网络，设置优化器和损失函数

    if args.resume:
        generator.load_state_dict(torch.load(args.resume)['generator'])
        discriminator.load_state_dict(torch.load(args.resume)['discriminator'])
        optimizer_G.load_state_dict(torch.load(args.resume)['optimizer_G'])
        optimizer_D.load_state_dict(torch.load(args.resume)['optimizer_D'])
        print(f'Pretrained models have been loaded.\n')
    else:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        print('Learning from scratch.')

    if use_cuda:
        torch.cuda.set_device(args.gpu)

        generator = generator.cuda()
        discriminator = discriminator.cuda()

    else:
        print('Using CPU, this will be slow')

    trainloader = DataLoader(dataset=SequenceDataset(channels=args.channels, size=args.size, videos_dir=args.videos_dir, time_steps=args.time_steps), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    #注意，dataset是一个SequenceDataset类，而trainloader是一个Dataloader类
    #二者关系是先自己定义一个数据类（是Dataset的子类），然后用dataloader封装数据实现迭代
    #PyTorch 数据加载实用程序的核心是 torch.utils.data.DataLoader类。它表示在数据集上可迭代的 Python
    #首先要创建一个数据集类，将文件夹中的图片转换成可供索引的Sequencedataset类，它是dataset类的子类
    #然后调用torch.utils.data.DataLoader()构建一个可迭代的类加载程序
    generator.train()
    discriminator.train()

    with torch.set_grad_enabled(True):
        for ep in range(args.epochs):
        #大循环，epochs的个数
            g_loss_sum = 0
            d_loss_sum = 0

            for i, clips in enumerate(trainloader):
            #中循环，迭代trainloader,以clips为单位，循环16次,i=15时遍历结束
                # print(len(clips))  
                #这里clips为一个列表，里面装了5个元素，均为torch.Size([1，15, 256, 256])的张量
                #即通过一个selected_idx，分别包含了5个随机数为起点的连续5帧图像的信息
                # print(clips[0].shape)
                pbar = tqdm(clips)
                #clips现在是一个长度为5的可迭代对象,迭代元素为torch.Size([1，15, 256, 256])的张量
                #每一个迭代元素张量代表连续5帧图像的信息

                for j, frames in enumerate(pbar):
                #小循环，每一个frames代表连续5帧图像的信息，当j=4时遍历完成，
                # 即每次trainloader的返回(pbar)代表的是5个连续五帧图像
                    inputs = frames[:, 0:args.channels * (args.time_steps - 1), :, :]
                    last = frames[:, args.channels * (args.time_steps - 2):args.channels * (args.time_steps -1), :, :]
                    target = frames[:, args.channels * (args.time_steps -1):args.channels * args.time_steps, :, :]
                    #为什么这么选imputs和target，我是不懂得

                    # print(inputs.shape,last.shape,target.shape)
                    #三者的shape分别为[1, 12, 256, 256]，[1, 3, 256, 256]，[1, 3, 256, 256]

                    if use_cuda:
                        inputs = inputs.cuda()
                        last = last.cuda()
                        target = target.cuda()
                        # print(target)

                    generated = generator(inputs) #前向传播

                    # print(generated.shape)
                    #丢入生成器得到的输出为[1, 3, 256, 256]，与target相同

                    d_t = discriminator(target)
                    d_g = discriminator(generated)

                    if use_cuda:
                        d_t = d_t.cuda()
                        d_g = d_g.cuda()
                    # print(generated.device)
                    # print(target.device)
                    d_loss = discriminator_loss(d_t, d_g)
                    optimizer_D.zero_grad()
                    d_loss.backward(retain_graph=True)
                    optimizer_D.step()

                    int_loss = intensity_loss(generated, target) 
                    grad_loss = gradient_loss(generated, target)
                    adv_loss = adversarial_loss(d_g)
                    
                    g_loss = int_loss + grad_loss+0.05 * adv_loss
                    # print(type(g_loss)) #计算损失，得到的都是一个tensor数值

                    optimizer_G.zero_grad()
                    g_loss.backward()
                    optimizer_G.step()
                    #每完成一次小循环(即利用五个不同的随机起始点得到的连续五帧图像，一个Train计算一次损失函数
                    #这里有个疑问，为什么不把它移到里面去，这样不是只有最后一个frames，即pbar的最后一次迭代得到了利用吗
                
                    if use_cuda:
                        torch.cuda.synchronize()  #计时函数

                    d_loss_sum += d_loss.item()
                    g_loss_sum += g_loss.item()
                    #item()返回的是一个浮点型数据，所以我们在求loss或者accuracy时，一般使用item()，而不是直接取它对应的元素

                    if j == 0:
                        diff_map = torch.sum(torch.abs(generated - target)[0], 0)
                        diff_map -= diff_map.min()
                        diff_map /= diff_map.max()
                        diff_map *= 255
                        diff_map = diff_map.detach().cpu().numpy().astype('uint8')
                        #当需要进一步运算时，避免计算梯度，需要detach()阻断反向传播
                        #而此时变量仍然在GPU上，即显存上，内存操作可能会找不到该变量，因此调用.cpu()
                        #将tensor数据转化为numpy类型
                        heat_map = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
                        cv2.imwrite(os.path.join('D:\\Study\\SRTP\\futureframe_base',args.save_logs_dir, f'{ep}_{i}_{time()}.jpg'), heat_map)
                    
                    pbar.set_postfix(_1_Epoch=f'{ep+1}/{args.epochs}',
                                    _2_int_loss=f'{int_loss:.5f}',
                                    _3_grad_loss=f'{grad_loss:.5f}',
                                    _4_adv_loss=f'{adv_loss:.5f}',
                                     _5_gen_loss=f'{g_loss.item():.5f}',
                                     _6_dis_loss=f'{d_loss.item():.5f}'
                                    )

            # print(len(clips),len(trainloader)) #trainloader=16,clips=5

            g_loss_mean = g_loss_sum / (len(clips) * len(trainloader))
            d_loss_mean = d_loss_sum / (len(clips) * len(trainloader))
            print('G Loss: ', g_loss_mean)
            print('D Loss: ', d_loss_mean)

            if(ep + 1) % args.save_per_epoch == 0:  #此时代表阶段训练完成,每5次保存训练模型
                model_dict = {'generator': generator.state_dict(), 'optimizer_G': optimizer_G.state_dict(),
                              'discriminator': discriminator.state_dict(), 'optimizer_D': optimizer_D.state_dict()}
                # torch.save(model_dict, os.path.join(args.save_model_dir, f'ckpt_{ep + 1}_{g_loss_mean}.pth'))
                torch.save(model_dict, os.path.join('saved_model',f'ckpt_{ep + 1}_{g_loss_mean}_{d_loss_mean}.pth'))
             
if __name__ == "__main__":
    train()