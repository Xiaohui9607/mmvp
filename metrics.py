import torch
from skimage.measure import compare_ssim
from torch.nn import functional as F


def peak_signal_to_noise_ratio(true, pred):
    return 10.0 * torch.log(torch.tensor(1.0) / F.mse_loss(true, pred)) / torch.log(torch.tensor(10.0))

def mse_to_psnr(mse):
    return (10.0 * torch.log(torch.tensor(1.0) / mse) / torch.log(torch.tensor(10.0))).numpy()


def calc_ssim(ground_truth, target):
    (score, diff) = compare_ssim(ground_truth, target, full=True)
    diff = (diff*255).astype("uint8")
    return score, diff
