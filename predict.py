import os
import glob
import numpy as np
import torch
import PIL.Image
from options import Options
from model import Model
from torchvision.transforms import functional as F

opt = Options().parse()


def save_to_local(tensor_list, folder):
    for idx_, tensor in enumerate(tensor_list):
        img = F.to_pil_image(tensor.squeeze().detach().cpu())
        img.save(os.path.join(folder, "predict_%s.jpg" % idx_))


def predict(net, data, save_path=None):
    images = [F.to_tensor(F.resize(F.to_pil_image(im), (opt.height, opt.width))).unsqueeze(0).to(opt.device)
              for im in torch.from_numpy(data).unbind(0)]
    actions = torch.zeros(1, len(images), 5).to(opt.device)
    states = torch.zeros_like(actions).to(opt.device)
    actions = actions.permute([1, 0, 2]).unbind(0)
    states = states.permute([1, 0, 2]).unbind(0)

    with torch.no_grad():
        gen_images, gen_states = net(images, actions, states[0])
        save_images = images[:opt.context_frames] + gen_images[opt.context_frames-1:]
        # save_images = images
        if save_path:
            save_to_local(save_images, save_path)


def smaple_to_npy(path):
    files = sorted(glob.glob(os.path.join(path, '*.jpg')))
    imglist = []
    for file in files:
        img = np.array(PIL.Image.open(file)).transpose([2, 0, 1])[np.newaxis, ...]
        imglist.append(img)
    img = np.concatenate(imglist, axis=0)
    return img


if __name__ == '__main__':
    images = smaple_to_npy("/home/golf/data/CY101/can_coke/trial_1/exec_1/push")
    opt.schedsamp_k = -1
    m = Model(opt)
    m.load_weight("./weight/net_epoch_30.pth")

    net = m.net

    predict(net, images, save_path="predict/")




