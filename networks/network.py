import math
import torch
from torch import nn
from torch.nn import functional as F
from .layer import ConvLSTM, haptic_feat, audio_feat, vibro_feat, haptic_head, audio_head, vibro_head

RELU_SHIFT = 1e-12
HAPTIC_DIM = [48, 10]
AUDIO_DIM = [16, 513]


class network(nn.Module):
    def __init__(self,
                 opt,
                 channels=3,
                 height=64,
                 width=64,
                 iter_num=-1.0,
                 k=-1,
                 num_masks=10,
                 stp=False,
                 cdna=True,
                 dna=False,
                 context_frames=2,
                 DNA_KERN_SIZE = 5,
                 HAPTIC_LAYER = 16,
                 BEHAVIOR_LAYER = 9,
                 AUDIO_LAYER = 16,
                 VIBRO_LAYER = 16
                 ):
        super(network, self).__init__()
        if stp + cdna + dna != 1:
            raise ValueError('More than one, or no network option specified.')
        self.lstm_size = [32, 32, 64, 64, 128, 64, 32]
        self.lstm_size = [l//2 for l in self.lstm_size]   # ligthen network
        self.dna = dna
        self.stp = stp
        self.cdna = cdna
        self.channels = channels
        self.num_masks = num_masks
        self.height = height
        self.width = width
        self.context_frames = context_frames
        self.k = k
        self.iter_num = iter_num
        self.opt = opt
        self.RELU_SHIFT = RELU_SHIFT

        # self.STATE_DIM = STATE_DIM
        self.DNA_KERN_SIZE = DNA_KERN_SIZE
        self.HAPTIC_DIM = HAPTIC_DIM
        self.AUDIO_LAYER = AUDIO_LAYER
        self.HAPTIC_LAYER = HAPTIC_LAYER
        self.BEHAVIOR_LAYER = BEHAVIOR_LAYER
        self.VIBRO_LAYER = VIBRO_LAYER
        # N * 3 * H * W -> N * 32 * H/2 * W/2
        self.enc0 = nn.Conv2d(in_channels=channels, out_channels=self.lstm_size[0], kernel_size=5, stride=2, padding=2)
        self.enc0_norm = nn.BatchNorm2d(self.lstm_size[0])

        # N * 32 * H/2 * W/2 -> N * 32 * H/2 * W/2
        self.lstm1 = ConvLSTM(in_channels=self.lstm_size[0], out_channels=self.lstm_size[0], kernel_size=5, padding=2)
        self.lstm1_norm = nn.BatchNorm2d(self.lstm_size[0])

        # N * 32 * H/2 * W/2 -> N * 32 * H/2 * W/2
        self.lstm2 = ConvLSTM(in_channels=self.lstm_size[0], out_channels=self.lstm_size[1], kernel_size=5, padding=2)
        self.lstm2_norm = nn.BatchNorm2d(self.lstm_size[1])


        # N * 32 * H/4 * W/4 -> N * 32 * H/4 * W/4
        self.enc1 = nn.Conv2d(in_channels=self.lstm_size[1], out_channels=self.lstm_size[1], kernel_size=3, stride=2, padding=1)
        # N * 32 * H/4 * W/4 -> N * 64 * H/4 * W/4
        self.lstm3 = ConvLSTM(in_channels=self.lstm_size[1], out_channels=self.lstm_size[2], kernel_size=5, padding=2)
        self.lstm3_norm = nn.BatchNorm2d(self.lstm_size[2])

        # N * 64 * H/4 * W/4 -> N * 64 * H/4 * W/4
        self.lstm4 = ConvLSTM(in_channels=self.lstm_size[2], out_channels=self.lstm_size[3], kernel_size=5, padding=2)
        self.lstm4_norm = nn.BatchNorm2d(self.lstm_size[3])


        # N * 64 * H/4 * W/4 -> N * 64 * H/8 * W/8
        self.enc2 = nn.Conv2d(in_channels=self.lstm_size[3], out_channels=self.lstm_size[3], kernel_size=3, stride=2, padding=1)

        # pass in state and action
        self.build_modalities_block()

        # N * (10+64) * H/8 * W/8 -> N * 64 * H/8 * W/8
        self.enc3 = nn.Conv2d(in_channels=self.lstm_size[3]+self.VIBRO_LAYER+self.HAPTIC_LAYER+self.AUDIO_LAYER+self.BEHAVIOR_LAYER,
                              out_channels=self.lstm_size[3], kernel_size=1, stride=1)

        # N * 64 * H/8 * W/8 -> N * 128 * H/8 * W/8
        self.lstm5 = ConvLSTM(in_channels=self.lstm_size[3], out_channels=self.lstm_size[4], kernel_size=5, padding=2)
        self.lstm5_norm = nn.BatchNorm2d(self.lstm_size[4])

        # N * 128 * H/8 * W/8 -> N * 128 * H/4 * W/4
        self.enc4 = nn.ConvTranspose2d(in_channels=self.lstm_size[4],
                                       out_channels=self.lstm_size[4],
                                       kernel_size=3, stride=2, output_padding=1, padding=1)
        # N * 128 * H/4 * W/4 -> N * 64 * H/4 * W/4
        self.lstm6 = ConvLSTM(in_channels=self.lstm_size[4], out_channels=self.lstm_size[5], kernel_size=5, padding=2)
        self.lstm6_norm = nn.BatchNorm2d(self.lstm_size[5])

        # N * 64 * H/4 * W/4 -> N * 64 * H/2 * W/2
        self.enc5 = nn.ConvTranspose2d(in_channels=self.lstm_size[5]+self.lstm_size[1],
                                       out_channels=self.lstm_size[5]+self.lstm_size[1],
                                       kernel_size=3, stride=2, output_padding=1, padding=1)
        # N * 64 * H/2 * W/2 -> N * 32 * H/2 * W/2
        self.lstm7 = ConvLSTM(in_channels=self.lstm_size[5]+self.lstm_size[1],
                              out_channels=self.lstm_size[6], kernel_size=5, padding=2)
        self.lstm7_norm = nn.BatchNorm2d(self.lstm_size[6])

        # N * 32 * H/2 * W/2 -> N * 32 * H * W
        self.enc6 = nn.ConvTranspose2d(in_channels=self.lstm_size[6]+self.lstm_size[0],
                                       out_channels=self.lstm_size[6], kernel_size=3,
                                       stride=2, output_padding=1, padding=1)
        self.enc6_norm = nn.BatchNorm2d(self.lstm_size[6])


        if self.dna:
            # N * 32 * H * W -> N * (DNA_KERN_SIZE*DNA_KERN_SIZE) * H * W
            self.enc7 = nn.ConvTranspose2d(in_channels=self.lstm_size[6], out_channels=self.DNA_KERN_SIZE**2, kernel_size=1, stride=1)
        else:
            # N * 32 * H * W -> N * 3 * H * W
            self.enc7 = nn.ConvTranspose2d(in_channels=self.lstm_size[6], out_channels=channels, kernel_size=1, stride=1)
            if self.cdna:
                # a reshape from lstm5: N * 128 * H/8 * W/8 -> N * (128 * H/8 * W/8)
                # N * (128 * H/8 * W/8) -> N * (10 * 5 * 5)
                in_dim = int(self.lstm_size[4] * self.height * self.width / 64)
                self.fc = nn.Linear(in_dim, self.DNA_KERN_SIZE * self.DNA_KERN_SIZE * self.num_masks)
            else:
                in_dim = int(self.lstm_size[4] * self.height * self.width / 64)
                self.fc = nn.Linear(in_dim, 100)
                self.fc_stp = nn.Linear(100, (self.num_masks-1) * 6)
        #  N * 32 * H * W -> N * 11 * H * W
        self.maskout = nn.ConvTranspose2d(self.lstm_size[6], self.num_masks+1, kernel_size=1, stride=1)
        # self.stateout = nn.Linear(STATE_DIM+ACTION_DIM, STATE_DIM)

    def build_modalities_block(self):
        if self.HAPTIC_LAYER != 0:
            self.haptic_feat = haptic_feat(self.HAPTIC_LAYER)
            self.haptic_head = haptic_head(self.lstm_size[3])

        if self.AUDIO_LAYER != 0:
            self.audio_feat = audio_feat(self.AUDIO_LAYER)
            self.audio_head = audio_head(self.lstm_size[3])

        if self.VIBRO_LAYER != 0:
            self.vibro_feat = vibro_feat(self.VIBRO_LAYER)
            self.vibro_head = vibro_head(self.lstm_size[3])

    def forward(self, images, haptics, audios, behaviors, vibros, train=True):
        '''
        :param inputs: T * N * C * H * W
        :param state: T * N * C
        :param action: T * N * C
        :return:
        '''

        lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None
        lstm_state5, lstm_state6, lstm_state7 = None, None, None
        haptic_feat_state, audio_feat_state, vibro_feat_state = None, None, None

        # haptic_feat_old = self.haptic_feat.feature(haptics[0]) if self.HAPTIC_LAYER != 0 else None
        # audio_feat_old = self.audio_feat.feature(audios[0]) if self.AUDIO_LAYER != 0 else None
        # vibro_feat_old = self.vibro_feat.feature(vibros[0]) if self.VIBRO_LAYER != 0 else None


        # haptic_feat_old, audio_feat_old = None, None

        gen_images = []
        gen_haptics = []
        gen_audios = []
        gen_vibros = []
        if self.k == -1 or not train:
            feedself = True
        else:
            num_ground_truth = round(images[0].shape[0] * (self.k / (math.exp(self.iter_num/self.k) + self.k)))
            feedself = False
            self.iter_num += 1

        for image, haptic, audio, vibro in zip(images[:-1], haptics[1:], audios[1:], vibros[1:]):

            done_warm_start = len(gen_images) >= self.context_frames

            if feedself and done_warm_start:
                # Feed in generated image.
                image = gen_images[-1]
            elif done_warm_start:
                # Scheduled sampling
                image = self.scheduled_sample(image, gen_images[-1], num_ground_truth)
            else:
                # Always feed in ground_truth
                image = image

            enc0 = self.enc0_norm(torch.relu(self.enc0(image)))

            lstm1, lstm_state1 = self.lstm1(enc0, lstm_state1)
            lstm1 = self.lstm1_norm(lstm1)

            lstm2, lstm_state2 = self.lstm2(lstm1, lstm_state2)
            lstm2 = self.lstm2_norm(lstm2)

            enc1 = torch.relu(self.enc1(lstm2))

            lstm3, lstm_state3 = self.lstm3(enc1, lstm_state3)
            lstm3 = self.lstm3_norm(lstm3)

            lstm4, lstm_state4 = self.lstm4(lstm3, lstm_state4)
            lstm4 = self.lstm4_norm(lstm4)

            enc2 = torch.relu(self.enc2(lstm4))

            # TODO: interaction + modalities feature extraction
            enc3, haptic_feat_state, audio_feat_state, vibro_feat_state = \
                self.interaction(enc2, haptic, audio, vibro, behaviors,
                                 haptic_feat_state, audio_feat_state, vibro_feat_state)


            gen_haptic = self.haptic_head(enc3) if self.HAPTIC_LAYER != 0 else None
            gen_audio = self.audio_head(enc3) if self.AUDIO_LAYER != 0 else None
            gen_vibro = self.vibro_head(enc3) if self.VIBRO_LAYER != 0 else None

            # TODO: done in modalities preiction
            gen_haptics.append(gen_haptic)
            gen_audios.append(gen_audio)
            gen_vibros.append(gen_vibro)

            # TODO: proceed on visual task
            lstm5, lstm_state5 = self.lstm5(enc3, lstm_state5)
            lstm5 = self.lstm5_norm(lstm5)
            enc4 = torch.relu(self.enc4(lstm5))

            lstm6, lstm_state6 = self.lstm6(enc4, lstm_state6)
            lstm6 = self.lstm6_norm(lstm6)
            # skip connection
            lstm6 = torch.cat([lstm6, enc1], dim=1)

            enc5 = torch.relu(self.enc5(lstm6))

            lstm7, lstm_state7 = self.lstm7(enc5, lstm_state7)
            lstm7 = self.lstm7_norm(lstm7)
            # skip connection
            lstm7 = torch.cat([lstm7, enc0], dim=1)

            enc6 = self.enc6_norm(torch.relu(self.enc6(lstm7)))

            enc7 = torch.relu(self.enc7(enc6))

            if self.dna:
                if self.num_masks != 1:
                    raise ValueError('Only one mask is supported for DNA model.')
                transformed = [self.dna_transformation(image, enc7)]
            else:
                transformed = [torch.sigmoid(enc7)]
                _input = lstm5.view(lstm5.shape[0], -1)
                if self.cdna:
                    transformed += self.cdna_transformation(image, _input)
                else:
                    transformed += self.stp_transformation(image, _input)

            masks = torch.relu(self.maskout(enc6))
            masks = torch.softmax(masks, dim=1)
            mask_list = torch.split(masks, split_size_or_sections=1, dim=1)

            output = mask_list[0] * image
            for layer, mask in zip(transformed, mask_list[1:]):
                output += layer * mask

            gen_images.append(output)

        return gen_images, gen_haptics, gen_audios, gen_vibros

    def interaction(self, vis_feat, haptic, audio, vibro, behaviors, haptic_feat_state, audio_feat_state, vibro_feat_state):

        behavior_feat = behaviors.repeat(1, 1, vis_feat.shape[2] // behaviors.shape[2], vis_feat.shape[3] // behaviors.shape[3])
        enc2 = [vis_feat, behavior_feat]

        if not torch.allclose(haptic, torch.zeros_like(haptic)):
            haptic_feat, haptic_feat_state = self.haptic_feat(haptic, haptic_feat_state)
            enc2.append(haptic_feat)

        if not torch.allclose(audio, torch.zeros_like(audio)):
            audio_feat, audio_feat_state = self.audio_feat(audio, audio_feat_state)
            enc2.append(audio_feat)

        if not torch.allclose(vibro, torch.zeros_like(vibro)):
            vibro_feat, vibro_feat_state = self.vibro_feat(vibro, vibro_feat_state)
            enc2.append(vibro_feat)

        enc2 = torch.cat(enc2, dim=1)

        enc3 = torch.relu(self.enc3(enc2))

        return enc3, haptic_feat_state, audio_feat_state, vibro_feat_state
        # return enc2, \
        #        haptic_feat, haptic_feat_state, \
        #        audio_feat, audio_feat_state, \
        #        vibro_feat, vv_lstm_feat_state


    def stp_transformation(self, image, stp_input):
        identity_params = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32).unsqueeze(1).repeat(1, self.num_masks-1)

        stp_input = self.fc(stp_input)
        stp_input = self.fc_stp(stp_input)
        stp_input = stp_input.view(-1, 6, self.num_masks-1) + identity_params
        params = torch.unbind(stp_input, dim=-1)

        transformed = [F.grid_sample(image, F.affine_grid(param.view(-1, 3, 2), image.size())) for param in params]
        return transformed


    def cdna_transformation(self, image, cdna_input):
        batch_size, height, width = image.shape[0], image.shape[2], image.shape[3]

        cdna_kerns = self.fc(cdna_input)
        cdna_kerns = cdna_kerns.view(batch_size, self.num_masks, 1, self.DNA_KERN_SIZE, self.DNA_KERN_SIZE)
        cdna_kerns = torch.relu(cdna_kerns - self.RELU_SHIFT) + self.RELU_SHIFT
        norm_factor = torch.sum(cdna_kerns, dim=[2,3,4], keepdim=True)
        cdna_kerns /= norm_factor

        cdna_kerns = cdna_kerns.view(batch_size*self.num_masks, 1, self.DNA_KERN_SIZE, self.DNA_KERN_SIZE)
        image = image.permute([1, 0, 2, 3])

        transformed = torch.conv2d(image, cdna_kerns, stride=1, padding=[(self.DNA_KERN_SIZE-1)//2, (self.DNA_KERN_SIZE-1)//2], groups=batch_size)

        transformed = transformed.view(self.channels, batch_size, self.num_masks, height, width)
        transformed = transformed.permute([1, 0, 3, 4, 2])
        transformed = torch.unbind(transformed, dim=-1)

        return transformed

    def dna_transformation(self, image, dna_input):
        image_pad = F.pad(image, [2, 2, 2, 2, 0, 0, 0, 0], "constant", 0)
        height, width = image.shape[2], image.shape[3]

        inputs = []

        for xkern in range(self.DNA_KERN_SIZE):
            for ykern in range(self.DNA_KERN_SIZE):
                inputs.append(image_pad[:, :, xkern:xkern+height, ykern:ykern+width].clone().unsqueeze(dim=1))
        inputs = torch.cat(inputs, dim=4)

        kernel = torch.relu(dna_input-self.RELU_SHIFT)+self.RELU_SHIFT
        kernel = kernel / torch.sum(kernel, dim=1, keepdim=True).unsqueeze(2)

        return torch.sum(kernel*inputs, dim=1, keepdim=False)

    def scheduled_sample(self, ground_truth_x, generated_x, num_ground_truth):
        generated_examps = torch.cat([ground_truth_x[:num_ground_truth, ...], generated_x[num_ground_truth:, :]], dim=0)
        return generated_examps