from options import Options
from model import Model


def train():
    opt = Options().parse()
    opt.use_behavior = True
    opt.use_audio = True
    opt.use_vibro = True
    opt.use_haptic = True
    opt.aux = True
    print("Model Config: ",opt)
    model = Model(opt)
    model.train()

if __name__ == '__main__':
    train()