import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim


def mse_to_psnr(mse):
    return 10.0 * torch.log(torch.tensor(1.0) / mse) / torch.log(torch.tensor(10.0))


def calc_ssim(ground_truth, target):
    (score, diff) = compare_ssim(ground_truth, target, full=True)
    diff = (diff*255).astype("uint8")
    return score, diff

bitplane = [0.528331, 0.822151, 0.927822, 0.955548, 0.970555, 0.974374, 0.975792, 0.976546]
fibplane = [0.004993, 0.342038, 0.552443, 0.706182, 0.807419, 0.869442, 0.944113, 0.959702]
roberts = [0.583387, 0.583387, 0.583387, 0.583387, 0.583387, 0.583387, 0.583387, 0.583387]
prewitt = [0.618760, 0.618760, 0.618760, 0.618760, 0.618760, 0.618760, 0.618760, 0.618760]
canny = [0.768912, 0.768912, 0.768912, 0.768912, 0.768912, 0.768912, 0.768912, 0.768912]
sobel = [0.621173, 0.621173, 0.621173, 0.621173, 0.621173, 0.621173, 0.621173, 0.621173]

# if __name__ == '__main__':
#     import seaborn as sns
#
#     sns.set(style="darkgrid")
#     sns.lineplot(x=range(1,9), y=bitplane, legend='brief', label="bitplane")
#     sns.lineplot(x=range(1,9), y=fibplane, legend='brief', label="fibplane")
#     sns.lineplot(x=range(1,9), y=roberts, legend='brief', label="roberts")
#     sns.lineplot(x=range(1,9), y=canny, legend='brief', label="canny")
#     sns.lineplot(x=range(1,9), y=prewitt, legend='brief', label="prewitt")
#     sns.lineplot(x=range(1,9), y=sobel, legend='brief', label="sobel")
#     plt.xlabel('# pitplane')
#     plt.ylabel('ssim')
#     plt.savefig("all_ssim.png",dpi=300)
#     plt.show()

if __name__ == '__main__':
    import glob
    import cv2
    import os
    ground_truth_path = "/home/golf/code/data/ED/groundtruth"
    ground_truth_images = sorted(glob.glob(os.path.join(ground_truth_path, "*.jpg")),
                                 key=lambda x:int(x.split('_')[-1].split('.')[0]))
    reconstruction_paths = "/home/golf/code/data/ED/reconstructions"
    reconstruction_folders = glob.glob(os.path.join(reconstruction_paths, '*'))

    # read groundtruth images
    ground_truth_images = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY) for image_path in ground_truth_images]
    # for loop
    N = len(ground_truth_images)
    for reconstruction_folder in reconstruction_folders:
        ssim = 0
        # read recon images
        reconstruction_images = sorted(glob.glob(os.path.join(reconstruction_folder, "*")),
                                 key=lambda x:int(x.split('/')[-1].split('.')[0]))
        reconstruction_images = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY) for image_path in reconstruction_images]

        # for loop
        for recon_image, gt_image in zip(reconstruction_images, ground_truth_images):
            # compare ssim
            i_ssim, i_diff = calc_ssim(gt_image, recon_image)
            ssim += i_ssim

        # averging
        ssim /= N
        # print
        print(reconstruction_folder, ssim)