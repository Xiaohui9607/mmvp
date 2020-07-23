from options import Options
from model import Model


def train():
    opt = Options().parse()
    print("Model Config: ",opt)

    model = Model(opt)
    model.train()

if __name__ == '__main__':
    train()
