from options import Options
from model import Model

if __name__ == '__main__':
    opt = Options().parse()
    opt.data_dir = '../CY101NPY'
    opt.pretrained_model = '../net_epoch_0.pth'
    m = Model(opt)
    m.load_weight()
    m.evaluate(0)