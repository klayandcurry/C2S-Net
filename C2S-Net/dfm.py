from train_stage1 import *
import random
import scipy.io as scio
from utils.SNR_statistic import snr_calculate

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_model(datanoise, model, mask):
    model.eval()
    output1 = model.test_onepic(datanoise, mask)
    out = np.squeeze(output1)

    return out


def data_generate(data, cropsize=112, max_zero_ratio=0.4, max_attempts=100):
    h, x, y = np.shape(data)
    D = data
    attempts = 0
    while attempts < max_attempts:
        xx = cropsize
        hh = cropsize
        yy = random.randint(0, y - 1)
        xstar = random.randint(0, x - xx)
        hhstar = random.randint(0, h - hh)
        subdata = np.squeeze(D[hhstar:hhstar + hh, xstar:xstar + xx, yy])
        subdata = subdata.astype(np.float32)
        zero_ratio = np.sum(subdata == 0) / (cropsize * cropsize)
        if zero_ratio <= max_zero_ratio:
            break
        elif attempts+1 == max_attempts:
            print('error')
        attempts += 1
    subdata = subdata / np.max(np.abs(subdata))
    mask = np.zeros_like(subdata)
    h, w = np.shape(mask)
    num = np.random.randint(30, 71)  # HyperParameter
    pos = np.random.randint(0, w - num)
    mask[:, pos:pos + num] = 1

    return subdata, mask, pos, num


if __name__ == '__main__':
    print('load data')
    trainrootpath = r'your path'
    data = scio.loadmat(trainrootpath)['data']
    train_data = data[:, :, 0:800]
    val_data = data[:, :, 800:980]

    print("{:-^50s}".format("load model"))
    Fpath = r'your path'
    trainer = Trainer(opt, image_size=opt.image_size)
    trainer.load(Fpath)

    print("{:-^50s}".format("model construct completed"))
    save_path = 'your path'
    print("{:-^50s}".format("start test"))
    data_amount = 0
    total_number = 5000
    for i in range(1000000):
        subdata, mask, pos, num = data_generate(val_data)
        label = subdata
        test_input = subdata * (1 - mask)
        h, w = np.shape(test_input)
        test_input = test_input.reshape((1, 1, h, w)).astype(np.float32)
        test_input = torch.tensor(test_input)
        mask1 = mask.reshape((1, 1, h, w)).astype(np.float32)
        mask1 = torch.tensor(mask1)
        out = test_model(test_input, trainer, mask1)
        out = out * mask + label * (1 - mask)
        snr = snr_calculate(label, out)
        if 0.1 <= snr < 3.5:  # HyperParameter
            data_amount += 1
            np.save(save_path + 'val_data/' + 'val_data' + str(data_amount) + '.npy', out)
            np.save(save_path + 'val_label/' + 'val_label' + str(data_amount) + '.npy', label)
            np.save(save_path + 'val_mask/' + 'val_mask' + str(data_amount) + '.npy', mask)
            np.save(save_path + 'val_num/' + 'val_num' + str(data_amount) + '.npy', num)
            np.save(save_path + 'val_pos/' + 'val_pos' + str(data_amount) + '.npy', pos)
            if data_amount == total_number:
                print("*************************")
                break

