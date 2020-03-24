import os
import torch
from torch import nn
import matplotlib.pyplot as plt
from networks import network, baseline
from data import build_dataloader_CY101
from torch.nn import functional as F


def mse_to_psnr(mse):
    return 10.0 * torch.log(torch.tensor(1.0) / mse) / torch.log(torch.tensor(10.0))


def peak_signal_to_noise_ratio(true, pred):
    """Image quality metric based on maximal signal power vs. power of the noise.

    Args:
    true: the ground truth image.
    pred: the predicted image.
    Returns:
    peak signal to noise ratio (PSNR)
    """
    return 10.0 * torch.log(torch.tensor(1.0) / F.mse_loss(true, pred)) / torch.log(torch.tensor(10.0))


class Model():
    def __init__(self, opt):
        self.opt = opt
        self.device = self.opt.device
        print(opt.use_haptic, opt.use_behavior, opt.use_audio)
        train_dataloader, valid_dataloader = build_dataloader_CY101(opt)
        self.dataloader = {'train': train_dataloader, 'valid': valid_dataloader}
        if self.opt.baseline:
            self.net = baseline(self.opt, self.opt.channels, self.opt.height, self.opt.width, -1, self.opt.schedsamp_k,
                            self.opt.num_masks, self.opt.model=='STP', self.opt.model=='CDNA', self.opt.model=='DNA', self.opt.context_frames)
        else:
            self.net = network(self.opt, self.opt.channels, self.opt.height, self.opt.width, -1, self.opt.schedsamp_k,
                           self.opt.num_masks, self.opt.model=='STP', self.opt.model=='CDNA', self.opt.model=='DNA', self.opt.context_frames,
                           self.opt.dna_kern_size, self.opt.haptic_layer, self.opt.behavior_layer, self.opt.audio_layer)

        self.net.to(self.device)
        self.mse_loss = nn.MSELoss()

        if self.opt.pretrained_model:
            self.load_weight()
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.opt.learning_rate, weight_decay=1e-4)

    def train_epoch(self, epoch):
        print("--------------------start training epoch %2d--------------------" % epoch)
        for iter_, (images, haptics, audios, behaviors) in enumerate(self.dataloader['train']):
            self.net.zero_grad()
            if not self.opt.use_behavior:
                behaviors = torch.zeros_like(behaviors).to(self.device)
            if not self.opt.use_haptic:
                haptics = torch.zeros_like(haptics).to(self.device)
            if not self.opt.use_audio:
                audios = torch.zeros_like(audios).to(self.device)

            behaviors = behaviors.unsqueeze(-1).unsqueeze(-1)
            images = images.permute([1, 0, 2, 3, 4]).unbind(0)
            haptics = haptics.permute([1, 0, 2, 3, 4]).unbind(0)
            audios = audios.permute([1, 0, 2, 3, 4]).unbind(0)

            gen_images, gen_haptics, gen_audios = self.net(images, haptics, audios, behaviors)

            recon_loss, haptic_loss, audio_loss = 0.0, 0.0, 0.0
            loss, psnr = 0.0, 0.0

            for i, (image, gen_image) in enumerate(
                    zip(images[self.opt.context_frames:], gen_images[self.opt.context_frames-1:])):
                recon_loss += self.mse_loss(image, gen_image)
                psnr_i = peak_signal_to_noise_ratio(image, gen_image)
                psnr += psnr_i

            if gen_haptics is not None and self.opt.aux:
                # add gen haptic loss
                for i, (haptic, gen_haptic) in enumerate(
                        zip(haptics[self.opt.context_frames:], gen_haptics[self.opt.context_frames - 1:])):
                    haptic_loss += self.mse_loss(haptic, gen_haptic) * 1e-6
            if gen_audios is not None and self.opt.aux:
                # add gen audio loss
                for i, (audio, gen_audio) in enumerate(
                        zip(audios[self.opt.context_frames:], gen_audios[self.opt.context_frames - 1:])):
                    audio_loss += self.mse_loss(audio, gen_audio) * 1e-3

            recon_loss /= torch.tensor(self.opt.sequence_length - self.opt.context_frames)
            haptic_loss /= torch.tensor(self.opt.sequence_length - self.opt.context_frames)
            audio_loss /= torch.tensor(self.opt.sequence_length - self.opt.context_frames)

            loss = recon_loss + haptic_loss + audio_loss
            loss.backward()
            self.optimizer.step()

            if iter_ % self.opt.print_interval == 0:
                print("training epoch: %3d, iterations: %3d/%3d total_loss: %6f recon_loss: %6f haptic_loss： %6f audio_loss: %6f" %
                      (epoch, iter_, len(self.dataloader['train'].dataset)//self.opt.batch_size, loss, recon_loss, haptic_loss, audio_loss))

    def train(self):
        for epoch_i in range(0, self.opt.epochs):
            self.train_epoch(epoch_i)
            self.evaluate(epoch_i)
            self.save_weight(epoch_i)

    def evaluate(self, epoch, keep_frame=False):
        with torch.no_grad():
            if keep_frame:
                mse_loss = [0.0 for _ in range(self.opt.sequence_length - self.opt.context_frames)]
            else:
                mse_loss = 0.0
            for iter_, (images, haptics, audios, behaviors) in enumerate(self.dataloader['valid']):
                if not self.opt.use_haptic:
                    haptics = torch.zeros_like(haptics).to(self.device)
                if not self.opt.use_behavior:
                    behaviors = torch.zeros_like(behaviors).to(self.device)
                if not self.opt.use_audio:
                    audios = torch.zeros_like(audios).to(self.device)

                behaviors = behaviors.unsqueeze(-1).unsqueeze(-1)
                images = images.permute([1, 0, 2, 3, 4]).unbind(0)
                haptics = haptics.permute([1, 0, 2, 3, 4]).unbind(0)
                audios = audios.permute([1, 0, 2, 3, 4]).unbind(0)

                gen_images, _, _ = self.net(images, haptics, audios, behaviors, train=False)

                for i, (image, gen_image) in enumerate(
                        zip(images[self.opt.context_frames:], gen_images[self.opt.context_frames - 1:])):
                    if keep_frame:
                        mse_loss[i] += self.mse_loss(image, gen_image)
                    else:
                        mse_loss += self.mse_loss(image, gen_image)

            if keep_frame:
                mse_loss = [loss / len(self.dataloader['valid'].dataset) * self.opt.batch_size for loss in mse_loss]

            else:
                mse_loss /= (torch.tensor(self.opt.sequence_length - self.opt.context_frames) * len(self.dataloader['valid'].dataset)/self.opt.batch_size)
                print("evaluation epoch: %3d, recon_loss: %6f" % (epoch, mse_loss))
            return mse_loss


    def save_weight(self, epoch):
        torch.save(self.net.state_dict(), os.path.join(self.opt.output_dir, "net_epoch_%d.pth" % epoch))

    def load_weight(self, path=None):
        if path:
            self.net.load_state_dict(torch.load(path))
        elif self.opt.pretrained_model:
            self.net.load_state_dict(torch.load(self.opt.pretrained_model, map_location=torch.device('cpu')))