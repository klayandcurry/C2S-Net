import numpy as np
from utils.config import opt
from train_stage2 import confidence_mask
import os
from utils.SNR_statistic import snr_calculate
import scipy.io as scio
from Unet import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer(nn.Module):
    def __init__(self, opt, image_size=128):
        super(Trainer, self).__init__()
        self.image_size = image_size
        self.net1 = nn.DataParallel(Unet1(1, 1)).to(device)
        self.opt = opt
        self.l1loss = nn.L1Loss()
        self.mseloss = nn.MSELoss()

    def test_multistage(self, image):
        with torch.no_grad():
            output = self.net1(image.to(device))
        output = output.detach().cpu().numpy()
        return output

    def load(self, save_path):
        state_dict = torch.load(save_path)
        self.net1.load_state_dict(state_dict)
        return self


pos = 35
num = 40


def generate_mask2(data):
    mask = np.zeros_like(data)
    mask[:, pos:pos + num] = 1
    return mask


def test_model(datanoise, model, mask, stage):
    model.eval()
    output = model.test_multistage(datanoise)
    output = np.squeeze(output)
    rec = output
    return rec


def test_window(data, mask, model, stage):
    print("{:-^50s}".format("start test by window"))
    testdata_data = data
    h, w = np.shape(testdata_data)
    cropsize = 112  # HyperParameter
    step = 85  # HyperParameter
    numt = (h - cropsize - 1) // step + 2
    numh = (w - cropsize - 1) // step + 2
    result = np.zeros((h, w))
    weight = np.zeros((h, w))
    for tt in range(numt):
        print(tt)
        for hh in range(numh):
            if tt == 0:
                tstar = 0
            elif tt == numt - 1:
                tstar = h - cropsize
            else:
                tstar = step * tt
            if hh == 0:
                hstar = 0
            elif hh == numh - 1:
                hstar = w - cropsize
            else:
                hstar = step * hh
            subdata = testdata_data[tstar:tstar + cropsize, hstar:hstar + cropsize]
            d_max = np.max(np.abs(subdata))
            if d_max == 0:
                d_max = d_max + 0.000000001
                subdata = subdata / (d_max)
            else:
                subdata = subdata / (d_max)
            submask = mask[tstar:tstar + cropsize, hstar:hstar + cropsize]
            if stage == 1:
                subinput = subdata * (1 - submask)
            elif stage == 2:
                subinput = subdata

            submask1 = submask.reshape((1, 1, cropsize, cropsize)).astype(np.float32)
            subinput = subinput.reshape((1, 1, cropsize, cropsize)).astype(np.float32)
            submask1 = torch.tensor(submask1)
            subinput = torch.tensor(subinput)
            # ---------------------------
            if (d_max > 0):
                eval_result = test_model(subinput, model, submask1, stage)
                eval_result = eval_result * submask + subdata * (1 - submask)
                result[tstar:tstar + cropsize, hstar:hstar + cropsize] = result[tstar:tstar + cropsize,
                                                                         hstar:hstar + cropsize] + \
                                                                         eval_result * (d_max)
                weight[tstar:tstar + cropsize, hstar:hstar + cropsize] = weight[tstar:tstar + cropsize,
                                                                         hstar:hstar + cropsize] + \
                                                                         np.ones((cropsize, cropsize))
            else:
                weight[tstar:tstar + cropsize, hstar:hstar + cropsize] = weight[tstar:tstar + cropsize,
                                                                         hstar:hstar + cropsize] + \
                                                                         np.ones((cropsize, cropsize))
    rec = result / weight
    return rec


if __name__ == '__main__':
    opt._parse()
    print("{:-^50s}".format("load data"))
    path = r'your path'
    data = scio.loadmat(path)['data']
    label = data[:, :, 999]
    print("{:-^50s}".format("data construct completed"))

    # stage 1 model
    print("{:-^50s}".format("load model1"))
    Fpath1 = r"your path"
    trainer1 = Trainer(opt, image_size=opt.image_size)
    model1 = trainer1.load(Fpath1)
    print("{:-^50s}".format("model1 construct completed"))

    # stage 2 model
    print("{:-^50s}".format("load model2"))
    Fpath2 = r"your path"
    trainer2 = Trainer(opt, image_size=opt.image_size)
    model2 = trainer2.load(Fpath2)
    print("{:-^50s}".format("model2 construct completed"))

    mask = generate_mask2(label)
    miss = label * (1 - mask)

    # ---------------start test by window------------------
    result1 = test_window(label, mask, model1, stage=1)
    input2 = result1.copy()

    input2_miss = input2[:, pos:pos + num]
    label_miss = label[:, pos:pos + num]
    mask_miss = confidence_mask(input2_miss, label_miss)
    mask_stage2 = np.ones_like(input2)
    mask_stage2[:, pos:pos + num] = mask_miss

    input2 = input2 * mask_stage2
    result2 = test_window(input2, mask, model2, stage=2)

    SNR1 = round(snr_calculate(label, result1), 5)
    SNR2 = round(snr_calculate(label, result2), 5)
    print(SNR1, SNR2)
