from options import Options
from model import Model


def train():
    opt = Options().parse()
    print("Model Config: ",opt)
    # opt.use_haptic = True
    # opt.use_behavior = True
    # opt.use_audio = True
    # opt.aux = True
    model = Model(opt)
    model.train()

if __name__ == '__main__':
    train()