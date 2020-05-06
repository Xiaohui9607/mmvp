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

        if self.opt.baseline:

            self.net = baseline(self.opt, self.opt.channels, self.opt.height, self.opt.width, -1, self.opt.schedsamp_k,
                            self.opt.num_masks, self.opt.model=='STP', self.opt.model=='CDNA', self.opt.model=='DNA', self.opt.context_frames)
        else:
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
                # add gen haptic loss
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

    def evaluate(self, epoch, keep_frame=False, save_prediction=False):
        with torch.no_grad():
            if keep_frame:
                mse_loss = [0.0 for _ in range(self.opt.sequence_length - self.opt.context_frames)]
            else:
                mse_loss = 0.0
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

                gen_images, _, _, _ = self.net(images, haptics, audios, behaviors, vibros, train=False)

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

    def predict(self, files):
        with torch.no_grad():
            import re
            import numpy as np
            from data import generate_npy_vibro,generate_npy_haptic,generate_npy_audio,generate_npy_vision, BEHAVIORS
            datas = []
            ret = []
            gt = []
            for file in files:
                vision = file['vision']
                haptic1 = os.path.join(re.sub(r'vision_data_part[1-4]', 'rc_data', vision), 'proprioception', 'ttrq0.txt')
                haptic2 = os.path.join(re.sub(r'vision_data_part[1-4]', 'rc_data', vision), 'proprioception', 'cpos0.txt')
                audio = os.path.join(re.sub(r'vision_data_part[1-4]', 'rc_data', vision), 'hearing', '*.wav')
                vibro = os.path.join(re.sub(r'vision_data_part[1-4]', 'rc_data', vision), 'vibro', '*.tsv')

                out_vision_npys, n_frames = generate_npy_vision(vision, vision.split('/')[-1], 20)
                out_audio_npys = generate_npy_audio(audio, n_frames, vision.split('/')[-1], 20)
                out_haptic_npys, bins = generate_npy_haptic(haptic1, haptic2, n_frames, vision.split('/')[-1], 20)
                out_vibro_npys = generate_npy_vibro(vibro, n_frames, bins, vision.split('/')[-1], 20)

                out_behavior_npys = np.zeros(len(BEHAVIORS))
                out_behavior_npys[BEHAVIORS.index(vision.split('/')[-1])] = 1
                out_behavior_npys = torch.from_numpy(out_behavior_npys).float().to(self.device)

                for out_vision_npy, out_haptic_npy, out_audio_npy, out_vibro_npy in \
                        zip(out_vision_npys, out_haptic_npys, out_audio_npys, out_vibro_npys):
                    out_vision_npy = torch.from_numpy(out_vision_npy)
                    out_vision_npy = self.dataloader['valid'].dataset.image_transform(out_vision_npy).to(self.device)

                    out_haptic_npy = torch.from_numpy(out_haptic_npy).float()
                    out_haptic_npy = self.dataloader['valid'].dataset.haptic_transform(out_haptic_npy).to(self.device)

                    out_audio_npy = torch.from_numpy(out_audio_npy)
                    out_audio_npy = self.dataloader['valid'].dataset.audio_transform(out_audio_npy).to(self.device)

                    out_vibro_npy = torch.from_numpy(out_vibro_npy)
                    out_vibro_npy = self.dataloader['valid'].dataset.vibro_transform(out_vibro_npy).to(self.device)

                    datas.append([out_vision_npy, out_haptic_npy, out_audio_npy, out_behavior_npys, out_vibro_npy])

            for iter_, (images, haptics, audios, behaviors, vibros) in enumerate(datas):
                if not self.opt.use_haptic:
                    haptics = torch.zeros_like(haptics).to(self.device)
                if not self.opt.use_behavior:
                    behaviors = torch.zeros_like(behaviors).to(self.device)
                if not self.opt.use_audio:
                    audios = torch.zeros_like(audios).to(self.device)
                if not self.opt.use_vibro:
                    vibros = torch.zeros_like(vibros).to(self.device)

                behaviors = behaviors.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
                images = images.unsqueeze(0).permute([1, 0, 2, 3, 4]).unbind(0)
                haptics = haptics.unsqueeze(0).permute([1, 0, 2, 3, 4]).unbind(0)
                audios = audios.unsqueeze(0).permute([1, 0, 2, 3, 4]).unbind(0)
                vibros = vibros.unsqueeze(0).permute([1, 0, 2, 3, 4]).unbind(0)

                gen_images, _, _, _ = self.net(images, haptics, audios, behaviors, vibros, train=False)
                gt.append(images)
                ret.append(gen_images)
            return ret, gt