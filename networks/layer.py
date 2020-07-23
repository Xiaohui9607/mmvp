import torch
from torch import nn

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, forget_bias=1.0, padding=0):
        super(ConvLSTM, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=out_channels + in_channels, out_channels=4 * out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.forget_bias = forget_bias

    def forward(self, inputs, states):
        if states is None:
            states = (torch.zeros([inputs.shape[0], self.out_channels, inputs.shape[2], inputs.shape[3]], device=inputs.device),
                      torch.zeros([inputs.shape[0], self.out_channels, inputs.shape[2], inputs.shape[3]], device=inputs.device))
        if not isinstance(states, tuple):
            raise TypeError('states type is not right')

        c, h = states
        if not (len(c.shape) == 4 and len(h.shape) == 4 and len(inputs.shape) == 4):
            raise TypeError('')

        inputs_h = torch.cat((inputs, h), dim=1)
        i_j_f_o = self.conv(inputs_h)
        i, j, f, o = torch.split(i_j_f_o,  self.out_channels, dim=1)

        new_c = c * torch.sigmoid(f + self.forget_bias) + torch.sigmoid(i) * torch.tanh(j)
        new_h = torch.tanh(new_c) * torch.sigmoid(o)

        return new_h, (new_c, new_h)


class Repeat(nn.Module):
    def __init__(self, dim, n_repeat):
        super(Repeat, self).__init__()
        self.dim = dim
        self.n_repeat = n_repeat

    def forward(self, hf):
        shapes = [1 for _ in range(len((hf.shape)))]
        shapes[self.dim] = self.n_repeat
        return hf.repeat(shapes)


class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, hf):
        return hf.permute(self.dims)


class haptic_feat(nn.Module):
    def __init__(self, HAPTIC_LAYER):
        super(haptic_feat, self).__init__()
        self.HAPTIC_LAYER = HAPTIC_LAYER
        self.feature = nn.Sequential()
        self.feature.add_module('repeat', Repeat(dim=1, n_repeat=48))
        self.feature.add_module('permute', Permute([0, 3, 1, 2]))

        self.feature.add_module('hconv_1', nn.Conv2d(10, 16, 5, stride=2))
        self.feature.add_module('hrelu_1', nn.ReLU())
        self.feature.add_module('hbn_1', nn.BatchNorm2d(16))

        self.feature.add_module('hconv_2', nn.Conv2d(16, 16, 3, stride=2))
        self.feature.add_module('hrelu_2', nn.ReLU())
        self.feature.add_module('hbn_2', nn.BatchNorm2d(16))

        self.feature.add_module('hconv_3', nn.Conv2d(16, self.HAPTIC_LAYER, 3, stride=1))
        self.feature.add_module('hrelu_3', nn.ReLU())
        self.feature.add_module('hbn_3', nn.BatchNorm2d(self.HAPTIC_LAYER))

        self.feature_lstm = ConvLSTM(in_channels=self.HAPTIC_LAYER, out_channels=self.HAPTIC_LAYER, kernel_size=5,
                                     padding=2)

    def forward(self, haptic, state):
        haptic_feat = self.feature(haptic)
        haptic_feat, hv_lstm_feat_state = self.feature_lstm(haptic_feat, state)
        return haptic_feat, hv_lstm_feat_state


class audio_feat(nn.Module):
    def __init__(self, AUDIO_LAYER):
        super(audio_feat, self).__init__()
        self.AUDIO_LAYER = AUDIO_LAYER
        self.feature = nn.Sequential()
        self.feature.add_module('audio_1', nn.Conv2d(1, 8, [1, 3], stride=[1, 2], padding=[0, 1]))
        self.feature.add_module('arelu_1', nn.ReLU())
        self.feature.add_module('abn_1', nn.BatchNorm2d(8))

        self.feature.add_module('audio_2', nn.Conv2d(8, 8, [1, 3], stride=[1, 2], padding=[0, 1]))
        self.feature.add_module('arelu_2', nn.ReLU())
        self.feature.add_module('abn_2', nn.BatchNorm2d(8))

        self.feature.add_module('audio_3', nn.Conv2d(8, 16, [1, 3], stride=[1, 2], padding=[0, 1]))
        self.feature.add_module('arelu_3', nn.ReLU())
        self.feature.add_module('abn_3', nn.BatchNorm2d(16))

        self.feature.add_module('audio_4', nn.Conv2d(16, self.AUDIO_LAYER, [3, 3], stride=[2, 2], padding=[1, 1]))
        self.feature.add_module('arelu_4', nn.ReLU())
        self.feature.add_module('abn_4', nn.BatchNorm2d(self.AUDIO_LAYER))

        self.feature_lstm = ConvLSTM(in_channels=self.AUDIO_LAYER, out_channels=self.AUDIO_LAYER, kernel_size=5,
                                     padding=2)

    def forward(self, audio, state):
        audio_feat = self.feature(audio)
        audio_feat, av_lstm_feat_state = self.feature_lstm(audio_feat, state)
        return audio_feat, av_lstm_feat_state


class vibro_feat(nn.Module):
    def __init__(self, VIBRO_LAYER):
        super(vibro_feat, self).__init__()
        self.VIBRO_LAYER = VIBRO_LAYER
        self.feature = nn.Sequential()
        self.feature.add_module('repeat', Repeat(dim=1, n_repeat=128))
        self.feature.add_module('permute', Permute([0, 3, 1, 2]))

        self.feature.add_module('vconv_1', nn.Conv2d(3, 16, 5, stride=2, padding=2))
        self.feature.add_module('vrelu_1', nn.ReLU())
        self.feature.add_module('vbn_1', nn.BatchNorm2d(16))

        self.feature.add_module('vconv_2', nn.Conv2d(16, 16, 3, stride=1, padding=1))
        self.feature.add_module('vrelu_2', nn.ReLU())
        self.feature.add_module('vbn_2', nn.BatchNorm2d(16))
        self.feature.add_module('vmaxpool_1', nn.MaxPool2d(2))

        self.feature.add_module('vconv_3', nn.Conv2d(16, self.VIBRO_LAYER, 3, stride=2, padding=1))
        self.feature.add_module('vrelu_3', nn.ReLU())
        self.feature.add_module('vbn_3', nn.BatchNorm2d(self.VIBRO_LAYER))
        self.feature.add_module('vmaxpool_2', nn.MaxPool2d(2))

        self.feature_lstm = ConvLSTM(in_channels=self.VIBRO_LAYER, out_channels=self.VIBRO_LAYER, kernel_size=5,
                                     padding=2)

    def forward(self, vibro, state):
        vibro_feat = self.feature(vibro)
        vibro_feat, vv_lstm_feat_state = self.feature_lstm(vibro_feat, state)
        return vibro_feat, vv_lstm_feat_state


# Haptic head
class haptic_head(nn.Module):
    def __init__(self, in_channels):
        super(haptic_head, self).__init__()
        self.head = nn.Sequential()
        self.head.add_module('deconv_1',
                             nn.ConvTranspose2d(in_channels, 16, [3, 3], stride=[1, 1]))
        self.head.add_module('relu_1', nn.ReLU())
        self.head.add_module('deconv_2', nn.ConvTranspose2d(16, 10, [3, 3], stride=[1, 1]))
        self.head.add_module('relu_2', nn.ReLU())
        self.head.add_module('deconv_3', nn.Conv2d(10, 10, [1, 1], stride=1))
        self.head.add_module('relu_3', nn.ReLU())
        self.head.add_module('global_pool', nn.AvgPool2d(12))
        self.head.add_module('permute', Permute([0, 2, 3, 1]))

    def forward(self, enc3):
        haptic = self.head(enc3)
        return haptic

# Audio head
class audio_head(nn.Module):
    def __init__(self, in_channels):
        super(audio_head, self).__init__()
        self.head = nn.Sequential()
        self.head.add_module('deconv_1', nn.ConvTranspose2d(in_channels, 16, [3, 3], stride=[2, 2], output_padding=1, padding=1))
        self.head.add_module('relu_1', nn.ReLU())
        self.head.add_module('deconv_2', nn.ConvTranspose2d(16, 16, [1, 3], stride=[1, 2], output_padding=[0, 1], padding=[0, 1]))
        self.head.add_module('relu_2', nn.ReLU())
        self.head.add_module('deconv_3', nn.ConvTranspose2d(16, 8, [1, 3], stride=[1, 2], output_padding=[0, 1], padding=[0, 1]))
        self.head.add_module('relu_3', nn.ReLU())
        self.head.add_module('deconv_4', nn.ConvTranspose2d(8, 1, [1, 3], stride=[1, 2], output_padding=[0, 1], padding=[0, 1]))
        self.head.add_module('relu_4', nn.ReLU())

    def forward(self, enc3):
        audio = self.head(enc3)
        return audio

# Vibro head
class vibro_head(nn.Module):
    def __init__(self, in_channels):
        super(vibro_head, self).__init__()
        self.head = nn.Sequential()
        self.head.add_module('deconv_1',
                             nn.ConvTranspose2d(in_channels, 32, [3, 3], stride=[2, 2], output_padding=1, padding=1))
        self.head.add_module('relu_1', nn.ReLU())
        self.head.add_module('deconv_2', nn.ConvTranspose2d(32, 16, [3, 3], stride=[2, 2], output_padding=[1, 1], padding=[1, 1]))
        self.head.add_module('relu_2', nn.ReLU())
        self.head.add_module('deconv_3', nn.ConvTranspose2d(16, 16, [3, 3], stride=[2, 2], output_padding=[1, 1], padding=[1, 1]))
        self.head.add_module('relu_3', nn.ReLU())
        self.head.add_module('deconv_4', nn.ConvTranspose2d(16, 3, [3, 3], stride=[2, 2], output_padding=[1, 1], padding=[1, 1]))
        self.head.add_module('relu_4', nn.ReLU())
        self.head.add_module('global_pool', nn.AvgPool2d((128, 1)))
        self.head.add_module('permute', Permute([0, 2, 3, 1]))

    def forward(self, enc3):
        vibro = self.head(enc3)
        return vibro