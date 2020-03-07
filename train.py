from options import Options
from model import Model


def train():
    opt = Options().parse()
    print("----------%s----------"%opt.output_dir)
    opt.use_haptic = True
    opt.use_behavior = True
    opt.use_audio = True
    model = Model(opt)
    model.train()

if __name__ == '__main__':
    train()