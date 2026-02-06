from utils.config import opt
import scipy.io as scio
from torch.utils.data import dataset
import random
import cv2
from tqdm import tqdm
import math
import os
import torch.optim as optim
import numpy as np
import time
from Unet import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Dataset(dataset.Dataset):

    def __init__(self, traindata, cropsize, flipw=0.5, mode='train', iternum=2000):
        self.flipw = flipw
        self.cropsize = cropsize
        self.data = traindata
        self.h, self.x, self.y = np.shape(traindata)
        self.num = iternum
        self.mode = mode
        random1 = np.random.RandomState(0)
        self.seed = random1.randint(100000, size=self.num)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        D = self.data
        h = self.h
        if self.mode == 'train':
            for i in range(100):
                xx = self.cropsize
                hh = self.cropsize
                xstar = random.randint(0, self.x - xx)
                yy = random.randint(0, self.y - 1)
                hhstar = random.randint(0, h - hh)
                data1 = np.squeeze(D[hhstar:hhstar + hh, xstar:xstar + xx, yy])
                if (np.sum(data1 == 0) / xx ** 2) < 0.5:
                    data = data1
                    break
            data = data.astype(np.float32)
        else:
            for i in range(100):
                xx = self.cropsize
                hh = self.cropsize
                random1 = np.random
                hhstar = random1.randint(0, h - self.cropsize)
                xstar = random1.randint(0, self.x - xx)
                yy = random1.randint(0, self.y - 1)
                data1 = np.squeeze(D[hhstar:hhstar + hh, xstar:xstar + xx, yy])
                if (np.sum(data1 == 0) / xx ** 2) < 0.5:
                    data = data1
                    break
            data = data.astype(np.float32)

        data = data / np.max(np.abs(data))

        flipw1 = np.random.uniform(0, 1)
        if flipw1 < self.flipw:
            data = data[:, ::-1].copy()

        mask = np.zeros_like(data)
        h, w = np.shape(data)
        num = np.random.randint(25, 66)
        pos = np.random.randint(0, w - num)
        mask[:, pos:pos + num] = 1
        image = data.reshape((1, self.cropsize, self.cropsize))
        mask_ori = mask.reshape((1, self.cropsize, self.cropsize))
        return image, mask_ori, pos, num


class Trainer(nn.Module):
    def __init__(self, opt, image_size=112):
        super(Trainer, self).__init__()
        self.image_size = image_size
        self.net = nn.DataParallel(Unet1(1, 1)).to(device)
        self.opt = opt
        self.l1loss = nn.L1Loss()
        self.mseloss = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_step, gamma=opt.lr_decay)

    def train_onebatch(self, image, mask):
        self.lr_scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        input = image * (1 - mask)
        input = input.to(device)
        label = image.to(device)
        mask = mask.to(device)
        output_img = self.net(input.to(device))
        final_loss = 0
        loss = self.mseloss(output_img, label)
        final_loss += 1 * loss
        self.optimizer.zero_grad()
        final_loss.backward()
        self.optimizer.step()

        return loss, lr, output_img, label

    def test_onepic(self, image, mask):
        with torch.no_grad():
            output = self.net(image.to(device))
        output = output.detach().cpu().numpy()
        return output

    def forward(self, image, mask, train=True):
        if train:
            return self.train_onebatch(image, mask)
        else:
            input = image * (1 - mask)
            output = self.net(input.to(device))
            oriimg = image.numpy()
            output = output.detach().cpu().numpy()
            s = np.sum(oriimg ** 2)
            n = np.sum((oriimg - output) * (oriimg - output))
            snr = math.log(s / n, 10)
            snr = snr * 10
            return snr, output

    def save(self, epoch, SNR, save_path=None, **kwargs):
        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'your path'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_
            save_path = save_path + '_%s' % timestr + '_' + str(epoch + 1) + '_' + str(SNR) + ".pth"
        torch.save(self.net.state_dict(), save_path)
        return save_path

    def load(self, save_path):
        state_dict = torch.load(save_path)
        self.net.load_state_dict(state_dict)
        return self


SAVEROOT = 'your path'
if not os.path.exists(SAVEROOT):
    os.makedirs(SAVEROOT)
end_epoch = opt.end_epoch


def test_model(dataloader, model, epoch, ifsave=False, test_num=100000, name='val/'):
    SNR = 0.0
    number = 0
    model.eval()
    dir = SAVEROOT + name + str(epoch + 1) + '/'
    if not os.path.exists(dir) and ifsave:
        os.makedirs(dir)
    val_label = 'your path'
    if not os.path.exists(val_label):
        os.makedirs(val_label)
    val_input = 'your path'
    if not os.path.exists(val_input):
        os.makedirs(val_input)
    for idx, (img, mask, pos, num) in enumerate(dataloader):
        snr, outputimg = model(img, mask, train=False)

        SNR += snr
        number += 1
        if ifsave and idx % 50 == 0:
            i = 0
            mask = mask[i][0].numpy()
            img = img[i][0].numpy()
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            cv2.imwrite(dir + 'oriimg' + str(idx) + '_' + '.jpg', img * 255)
            input = img * (1 - mask)
            cv2.imwrite(dir + 'input' + str(idx) + '_' + '.jpg', input * 255)
            outimg = outputimg[i][0]
            outimg = (outimg - np.min(outimg)) / (np.max(outimg) - np.min(outimg))
            cv2.imwrite(dir + 'outimg' + str(idx) + '_' + '.jpg', outimg * 255)
        if idx > test_num:
            break
    return {"SNR": round(SNR / number, 5)}


def train():
    opt._parse()
    trainer = Trainer(opt, image_size=opt.image_size)
    print('load data')
    path = r'your path'
    data = scio.loadmat(path)['data']
    train_data = data[:, :, 0:800]
    val_data = data[:, :, 800:980]
    train_dataset = Dataset(train_data, cropsize=112, flipw=0.5, mode='train', iternum=10000)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.train_batch_size,
                                                   shuffle=False,
                                                   num_workers=opt.num_workers)
    val_dataset = Dataset(val_data, cropsize=112, flipw=0.5, mode='test', iternum=1500)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=opt.test_batch_size,
                                                 num_workers=opt.num_workers,
                                                 shuffle=False)
    SNR = []
    LOSS = []
    for epoch in range(end_epoch):
        trainer.train()
        Loss = 0.0
        number = 1
        for idx, (img, mask, pos, num) in tqdm(enumerate(train_dataloader),
                                               total=len(train_dataloader)):
            loss, lr, output_img, label = trainer.train_onebatch(img, mask)
            Loss = Loss + loss.detach().cpu().numpy()
            number = number + 1
        print("[%02d/%02d] Total Loss: %.5f\n" % (epoch + 1, end_epoch, round(Loss / number, 7)))
        LOSS.append(round(Loss / number, 7))
        ifsave = False
        if (epoch + 1) == end_epoch or epoch % 10 == 0:
            ifsave = True
        eval_result = test_model(val_dataloader, trainer, epoch, ifsave=ifsave, test_num=opt.test_num)
        print('eval_snr: ', eval_result)
        if epoch + 1 > 40 or (epoch + 1) % 10 == 0:
            best_path = trainer.save(epoch, eval_result['SNR'], save_path=None)
            print("net save to %s !" % best_path)
        SNR.append(eval_result["SNR"])
        np.save(SAVEROOT + 'SNR_stage1.npy', SNR)
        np.save(SAVEROOT + 'LOSS_stage1.npy', LOSS)


if __name__ == '__main__':
    train()
