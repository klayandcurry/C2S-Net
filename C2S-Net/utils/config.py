from pprint import pprint


class Config:
    num_workers = 0
    image_size = 164

    # param for optimizer
    weight_decay = 0.0000005
    lr_decay = 0.1  # 0.5   #0.1  # 1e-3 -> 1e-4
    lr = 1e-3  # 1e-3
    lr_step = 30000

    model_name = 'nodowmsample_Unet'
    # training
    end_epoch = 50
    train_batch_size = 10  # 24#14         14  16  18
    test_batch_size = 1
    # visualization
    env = model_name  # visdom env
    port = 8097
    plot_every = 10  # vis every N iter

    test_num = 10000  # 10000
    # model
    savepath = "./result/3M15D_gap/mse/"

    def _parse(self):
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()
