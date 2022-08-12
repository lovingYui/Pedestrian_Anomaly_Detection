from distutils.ccompiler import gen_lib_options
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset
from unet import UNet

from tqdm import tqdm

import cv2
import imageio

class args():
    checkpoint = 'saved_model\\ckpt_10_13750.929541015625_0.184050502628088.pth'

    channels = 3
    size = 256
    videos_dir = 'D:\\Study\\SRTP\\UCSD_Anomaly_Dataset.v1p2\\UCSDped2\\Test\\Test003'
    time_steps = 5

    gpu = 0

def evaluate():

    generator = UNet(in_channels=args.channels * (args.time_steps - 1), out_channels=args.channels) #12入3出
    # print(type(generator))
    generator.load_state_dict(torch.load(args.checkpoint)['generator'])
    print(f'The pre-trained generator has been loaded from', args.checkpoint)
    
    testloader = DataLoader(dataset=TestDataset(channels=args.channels, size=args.size, videos_dir=args.videos_dir, time_steps=args.time_steps), batch_size=1, shuffle=False, num_workers=0)
    
    if torch.cuda.is_available() and args.gpu is not None:
        use_cuda = True
        torch.cuda.set_device(args.gpu)

        generator = generator.cuda()

    else:
        use_cuda = False
        print("using CPU, this will be slow")

    heatmaps = []
    originals = []
    ganmaps=[]
    with torch.no_grad():
        for i, datas in enumerate(tqdm(testloader)):

            # print(datas)
            # print(type(datas)) datas为一个len为2的列表，返回seqs和o_seqs

            frames, o_frames = datas[0], datas[1]

            # print(frames.shape)    torch.Size([1, 15, 256, 256])
            # print(o_frames[0].shape)      torch.Size([1, 256, 256, 3])
          
            generator.eval()
            inputs = frames[:, :args.channels * (args.time_steps - 1), :, :]  #torch.Size([1, 12, 256, 256])
            target = frames[:, args.channels * (args.time_steps - 1):, :, :]  #torch.Size([1, 3, 256, 256])
            # print(type(inputs))

            if use_cuda:
                inputs = inputs.cuda()
                target = target.cuda()

            generated = generator(inputs)  ##torch.Size([1, 3, 256, 256])

            if use_cuda:
                generated = generated.cuda()

            ganmap = torch.sum(torch.abs(generated).squeeze(),0)
            ganmap -= ganmap.min()
            ganmap /= ganmap.max()
            ganmap *=255
            ganmap = ganmap.detach().cpu().numpy().astype('uint8')
            #ganmap = cv2.applyColorMap(ganmap, cv2.COLORMAP_JET)
            ganmap = cv2.cvtColor(ganmap, cv2.COLOR_BGR2RGB)
            ganmaps.append(ganmap)

            diffmap = torch.sum(torch.abs(generated - target).squeeze(), 0)
            diffmap -= diffmap.min()
            diffmap /= diffmap.max()
            diffmap *= 255
            diffmap = diffmap.detach().cpu().numpy().astype('uint8')

            heatmap = cv2.applyColorMap(diffmap, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmaps.append(heatmap)

            original = o_frames[-1].squeeze().detach().cpu().numpy().astype('uint8')
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            originals.append(original)

    imageio.mimsave(f'results/ganmap.gif', ganmaps, fps =30)
    imageio.mimsave(f'results/heatmap.gif', heatmaps, fps=30)
    imageio.mimsave(f'results/original.gif', originals, fps=30)


if __name__ == "__main__":
    evaluate()