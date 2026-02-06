import os
import numpy as np
from tqdm import tqdm
from utils.config import opt
import torch.optim as optim
from Unet import *
from torch.utils.data import dataset
from glob import glob
import math
import time
import re

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def confidence_mask(A, B):
    A = np.array(A)
    B = np.array(B)
    if A.shape != B.shape:
        raise ValueError("error")
    rows, cols = A.shape
    error = A - B
    total_mse = np.mean(error ** 2)
    threshold = total_mse / cols
    column_mse = []
    for i in range(cols):
        temp_error = np.zeros_like(error)
        temp_error[:, i] = error[:, i]
        col_mse = np.mean(temp_error ** 2)
        column_mse.append(col_mse)
    column_mse = np.array(column_mse)
    column_mask = (column_mse <= threshold).astype(int)
    mask = np.tile(column_mask, (rows, 1))
    ones_count = np.sum(column_mask == 1)
    ones_ratio = ones_count / cols
    return mask


class Dataset1(dataset.Dataset):

    def __init__(self, data_dir, label_dir, mask_dir, pos_dir, num_dir, size=112):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.mask_dir = mask_dir
        self.pos_dir = pos_dir
        self.num_dir = num_dir
        self.datalist = []
        self.labellist = []
        self.masklist = []
        self.poslist = []
        self.numlist = []
        for m in data_dir:
            data_paths = glob(os.path.join(m, "*.npy"))
            data_paths.sort(key=natural_sort_key)
            self.datalist = self.datalist + data_paths
        for i in label_dir:
            label_paths = glob(os.path.join(i, "*.npy"))
            label_paths.sort(key=natural_sort_key)
            self.labellist = self.labellist + label_paths
        for j in mask_dir:
            mask_paths = glob(os.path.join(j, "*.npy"))
            mask_paths.sort(key=natural_sort_key)
            self.masklist = self.masklist + mask_paths
        for h in pos_dir:
            pos_paths = glob(os.path.join(h, "*.npy"))
            pos_paths.sort(key=natural_sort_key)
            self.poslist = self.poslist + pos_paths
        for k in num_dir:
            num_paths = glob(os.path.join(k, "*.npy"))
            num_paths.sort(key=natural_sort_key)
            self.numlist = self.numlist + num_paths
        self.size = size

    def __len__(self):
        return len(self.labellist)

    def __getitem__(self, idx):
        data = np.load(self.datalist[idx]).astype(np.float32)
        label = np.load(self.labellist[idx]).astype(np.float32)
        mask = np.load(self.masklist[idx]).astype(np.float32)
        pos = np.load(self.poslist[idx])
        num = np.load(self.numlist[idx])

        data_miss = data[:, pos:pos + num]
        label_miss = label[:, pos:pos + num]
        mask_miss = confidence_mask(data_miss, label_miss)
        mask_stage2 = np.ones_like(data)
        mask_stage2[:, pos:pos + num] = mask_miss
        data = data * mask_stage2

        data = data.reshape((1, self.size, self.size))
        label = label.reshape((1, self.size, self.size))
        mask_stage2 = mask_stage2.reshape((1, self.size, self.size))
        return data, label, mask_stage2, pos, num


class Trainer(nn.Module):
    def __init__(self, opt, image_size=112):
        super(Trainer, self).__init__()
        self.image_size = image_size
        self.net = nn.DataParallel(Unet1(1, 1)).to(device)
        self.opt = opt
        self.L1loss = nn.L1Loss()
        self.mseloss = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_step, gamma=opt.lr_decay)

    def train_onebatch(self, data, label, mask, pos, num):
        self.lr_scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        input = data.to(device)
        label = label.to(device)
        mask = mask.to(device)
        output_img = self.net(input)
        final_loss = self.mseloss(output_img, label)
        self.optimizer.zero_grad()
        final_loss.backward()
        self.optimizer.step()

        return final_loss, lr, output_img, label

    def forward(self, image, oriimg, mask, pos, num, train=True):
        if train:
            return self.train_onebatch(image, oriimg, mask, pos, num)

        else:
            output = self.net(image.to(device))
            oriimg = oriimg.numpy()
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

    print('start test')
    for idx, (data, label, mask, pos, num) in tqdm(enumerate(dataloader), total=len(dataloader)):
        img = data
        oriimg = label
        snr, outputimg = model(img, oriimg, mask, pos, num, train=False)
        SNR += snr
        number += 1
    return {"SNR": round(SNR / number, 5)}


def train():
    opt._parse()
    trainer = Trainer(opt, image_size=opt.image_size)
    print('load data')
    train_path = [r'your path']
    label_path = [r'your path']
    mask_path = [r'your path']
    pos_path = [r'your path']
    num_path = [r'your path']
    train_dataset = Dataset1(data_dir=train_path, label_dir=label_path, mask_dir=mask_path, pos_dir=pos_path,
                             num_dir=num_path)
    val_input_path = [r'your path']
    val_label_path = [r'your path']
    val_mask_path = [r'your path']
    val_pos_path = [r'your path']
    val_num_path = [r'your path']
    val_dataset = Dataset1(data_dir=val_input_path, label_dir=val_label_path, mask_dir=val_mask_path,
                           pos_dir=val_pos_path, num_dir=val_num_path)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.train_batch_size,
                                                   shuffle=False,
                                                   num_workers=opt.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=opt.test_batch_size,
                                                 num_workers=opt.num_workers,
                                                 shuffle=False)
    SNR = []
    LOSS = []
    for epoch in range(end_epoch):
        trainer.train()
        Loss = 0.0
        number = 0
        for idx, (data, label, mask, pos, num) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            loss, lr, output_img, label = trainer.train_onebatch(data, label, mask, pos, num)
            Loss = Loss + loss.detach().cpu().numpy()
            number = number + 1
        print("[%02d/%02d] Total Loss: %.5f\n" % (epoch + 1, end_epoch, round(Loss / number, 7)))
        LOSS.append(round(Loss / number, 7))
        ifsave = False
        if (epoch + 1) == end_epoch:
            ifsave = True
        eval_result = test_model(val_dataloader, trainer, epoch, ifsave=ifsave, test_num=opt.test_num)
        print('eval_snr: ', eval_result)
        SNR.append(eval_result["SNR"])
        if epoch + 1 > (end_epoch - 10) or (epoch + 1) % 10 == 0:
            best_path = trainer.save(epoch, eval_result['SNR'], save_path=None)
            print("net save to %s !" % best_path)
        np.save(SAVEROOT + 'SNR_stage2.npy', SNR)
        np.save(SAVEROOT + 'LOSS_stage2.npy', LOSS)


if __name__ == '__main__':
    train()
