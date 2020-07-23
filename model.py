import os
import torch
from torch import nn
import cv2
import numpy as np
from networks import network
from data import build_dataloader_CY101
from torch.nn import functional as F
from metrics import calc_ssim, mse_to_psnr, peak_signal_to_noise_ratio


class Model():
    def __init__(self, opt):
        self.opt = opt
        self.device = self.opt.device
        print("use haptic: ", opt.use_haptic, "    use behavior: ", opt.use_behavior, "    use audio: ", opt.use_audio, "    use vibro: ", opt.use_vibro)
        train_dataloader, valid_dataloader = build_dataloader_CY101(opt)
        self.dataloader = {'train': train_dataloader, 'valid': valid_dataloader}
        self.gen_images = []
        if not self.opt.use_haptic:
            self.opt.haptic_layer = 0

        if not self.opt.use_vibro:
            self.opt.vibro_layer = 0

        if not self.opt.use_audio:
            self.opt.audio_layer = 0


        self.net = network(self.opt, self.opt.channels, self.opt.height, self.opt.width, -1, self.opt.schedsamp_k,
                       self.opt.num_masks, self.opt.model=='STP', self.opt.model=='CDNA', self.opt.model=='DNA', self.opt.context_frames,
                       self.opt.dna_kern_size, self.opt.haptic_layer, self.opt.behavior_layer, self.opt.audio_layer, self.opt.vibro_layer)

        self.net.to(self.device)
        self.mse_loss = nn.MSELoss()

        if self.opt.pretrained_model:
            self.load_weight()
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.opt.learning_rate, weight_decay=1e-4)

    def train_epoch(self, epoch):
        print("--------------------start training epoch %2d--------------------" % epoch)
        for iter_, (images, haptics, audios, behaviors, vibros) in enumerate(self.dataloader['train']):
            self.net.zero_grad()
            if not self.opt.use_behavior:
                behaviors = torch.zeros_like(behaviors).to(self.device)
            if not self.opt.use_haptic:
                haptics = torch.zeros_like(haptics).to(self.device)
            if not self.opt.use_audio:
                audios = torch.zeros_like(audios).to(self.device)
            if not self.opt.use_vibro:
                vibros = torch.zeros_like(vibros).to(self.device)

            behaviors = behaviors.unsqueeze(-1).unsqueeze(-1)
            images = images.permute([1, 0, 2, 3, 4]).unbind(0)
            haptics = haptics.permute([1, 0, 2, 3, 4]).unbind(0)
            audios = audios.permute([1, 0, 2, 3, 4]).unbind(0)
            vibros = vibros.permute([1, 0, 2, 3, 4]).unbind(0)

            gen_images, gen_haptics, gen_audios, gen_vibros = self.net(images, haptics, audios, behaviors, vibros)

            recon_loss, haptic_loss, audio_loss, vibro_loss = 0.0, 0.0, 0.0, 0.0
            loss, psnr = 0.0, 0.0

            for i, (image, gen_image) in enumerate(
                    zip(images[self.opt.context_frames:], gen_images[self.opt.context_frames-1:])):
                recon_loss += self.mse_loss(image, gen_image)
                psnr_i = peak_signal_to_noise_ratio(image, gen_image)
                psnr += psnr_i

            if self.opt.use_haptic and self.opt.aux:
                for i, (haptic, gen_haptic) in enumerate(
                        zip(haptics[self.opt.context_frames:], gen_haptics[self.opt.context_frames - 1:])):
                    haptic = torch.mean(haptic, dim=[1,2], keepdim=True)
                    haptic_loss += self.mse_loss(haptic[..., -3:], gen_haptic[..., -3:]) * 1e-4
            if self.opt.use_audio and self.opt.aux:
                # add gen audio loss
                for i, (audio, gen_audio) in enumerate(
                        zip(audios[self.opt.context_frames:], gen_audios[self.opt.context_frames - 1:])):
                    audio_loss += self.mse_loss(audio, gen_audio) * 1e-3

            if self.opt.use_vibro and self.opt.aux:
                # add gen audio loss
                for i, (vibro, gen_vibro) in enumerate(
                        zip(vibros[self.opt.context_frames:], gen_vibros[self.opt.context_frames - 1:])):
                    vibro_loss += self.mse_loss(vibro, gen_vibro) * 1e-4

            recon_loss /= torch.tensor(self.opt.sequence_length - self.opt.context_frames)
            haptic_loss /= torch.tensor(self.opt.sequence_length - self.opt.context_frames)
            audio_loss /= torch.tensor(self.opt.sequence_length - self.opt.context_frames)
            vibro_loss /= torch.tensor(self.opt.sequence_length - self.opt.context_frames)

            loss = recon_loss + haptic_loss + audio_loss + vibro_loss
            loss.backward()
            self.optimizer.step()

            if iter_ % self.opt.print_interval == 0:
                print("training epoch: %3d, iterations: %3d/%3d total_loss: %6f recon_loss: %6f haptic_loss： %6f audio_loss: %6f vibro_loss: %6f" %
                      (epoch, iter_, len(self.dataloader['train'].dataset)//self.opt.batch_size, loss, recon_loss, haptic_loss, audio_loss, vibro_loss))

    def train(self):
        for epoch_i in range(0, self.opt.epochs):
            self.train_epoch(epoch_i)
            self.evaluate(epoch_i)
            self.save_weight(epoch_i)

    def evaluate(self, epoch, keep_frame=False, keep_batch=False, save_prediction=False, ssim=False):
        with torch.no_grad():
            loss = [[] for _ in range(self.opt.sequence_length - self.opt.context_frames)]
            for iter_, (images, haptics, audios, behaviors, vibros) in enumerate(self.dataloader['valid']):
                if not self.opt.use_haptic:
                    haptics = torch.zeros_like(haptics).to(self.device)
                if not self.opt.use_behavior:
                    behaviors = torch.zeros_like(behaviors).to(self.device)
                if not self.opt.use_audio:
                    audios = torch.zeros_like(audios).to(self.device)
                if not self.opt.use_vibro:
                    vibros = torch.zeros_like(vibros).to(self.device)

                behaviors = behaviors.unsqueeze(-1).unsqueeze(-1)
                images = images.permute([1, 0, 2, 3, 4]).unbind(0)
                haptics = haptics.permute([1, 0, 2, 3, 4]).unbind(0)
                audios = audios.permute([1, 0, 2, 3, 4]).unbind(0)
                vibros = vibros.permute([1, 0, 2, 3, 4]).unbind(0)

                gen_images, gen_haptics, gen_audios, gen_vibros = self.net(images, haptics, audios, behaviors, vibros, train=False)

                for i, (image, gen_image) in enumerate(
                        zip(images[self.opt.context_frames:], gen_images[self.opt.context_frames - 1:])):
                    stats = None
                    if ssim:
                        image = image.permute([0, 2, 3, 1]).unbind(0)
                        image = [cv2.cvtColor((im.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) for im in
                                 image]
                        gen_image = gen_image.permute([0, 2, 3, 1]).unbind(0)
                        gen_image = [cv2.cvtColor((im.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) for im
                                     in gen_image]
                        stats = [calc_ssim(im, gim)[0] for im, gim in zip(image, gen_image)]
                    else:
                        stats = mse_to_psnr(torch.mean((image-gen_image)**2, dim=[1,2,3]).cpu())

                    loss[i].extend(stats)

            if keep_frame:
                stds = [np.std(item) for item in loss]
                loss = [np.mean(item) for item in loss]
                return loss, stds
            if keep_batch:
                loss = np.stack([it for it in loss if it])
                loss = np.mean(loss, axis=0)
                return loss
            else:
                loss = np.stack(loss)
                loss = np.mean(loss)
                return loss

    def save_weight(self, epoch):
        torch.save(self.net.state_dict(), os.path.join(self.opt.output_dir, "net_epoch_%d.pth" % epoch))

    def load_weight(self, path=None):
        if path:
            self.net.load_state_dict(torch.load(path))
        elif self.opt.pretrained_model:
            self.net.load_state_dict(torch.load(self.opt.pretrained_model, map_location=torch.device('cpu')))
