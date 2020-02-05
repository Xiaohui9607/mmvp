import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from .split_data import split


IMG_EXTENSIONS = ('.npy',)


# def make_dataset(path):
    # image_folders = os.path.join(path)
    #
    # if not os.path.exists(image_folders):
    #     raise FileExistsError('some subfolders from data set do not exists!')

    # samples = []
    # for sample in os.listdir(image_folders):
    #     image  = os.path.join(image_folders, sample)
    #     samples.append(image)
    # return samples


def npy_loader(path):
    samples = torch.from_numpy(np.load(path))
    return samples


class CY101Dataset(Dataset):
    def __init__(self, files, image_transform=None, loader=npy_loader, device='cpu'):
        self.image_transform = image_transform
        self.samples = files
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 images, " + "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.loader = loader
        self.device = device

    def __getitem__(self, index):
        image = self.samples[index]
        image = self.loader(image)

        if self.image_transform is not None:
            image = torch.cat([self.image_transform(single_image).unsqueeze(0) for single_image in image.unbind(0)], dim=0)

        return image.to(self.device)

    def __len__(self):
        return len(self.samples)

def build_dataloader_CY101(opt):
    def crop(im):
        height, width = im.shape[1:]
        width = max(height, width)
        im = im[:, :width, :width]
        return im

    transform = transforms.Compose([
        transforms.Lambda(crop),
        transforms.ToPILImage(),
        transforms.Resize((opt.height, opt.width)),
        transforms.ToTensor()
    ])
    # TODO split data into train, valid, test, turn it in to file list
    trains, valids = split(opt.data_dir, ratio=0.9, strategy='instance')

    train_ds = CY101Dataset(
        # root=os.path.join(opt.data_dir+'/train'),
        files=trains,
        image_transform=transform,
        loader=npy_loader,
        device=opt.device
    )

    valid_ds = CY101Dataset(
        # root=os.path.join(opt.data_dir+'/valid'),
        files=valids,
        image_transform=transform,
        loader=npy_loader,
        device=opt.device
    )

    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    return train_dl, valid_dl

