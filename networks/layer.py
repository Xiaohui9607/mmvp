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
            raise TypeError("states type is not right")

        c, h = states
        if not (len(c.shape) == 4 and len(h.shape) == 4 and len(inputs.shape) == 4):
            raise TypeError("")

        inputs_h = torch.cat((inputs, h), dim=1)
        i_j_f_o = self.conv(inputs_h)
        i, j, f, o = torch.split(i_j_f_o,  self.out_channels, dim=1)

        new_c = c * torch.sigmoid(f + self.forget_bias) + torch.sigmoid(i) * torch.tanh(j)
        new_h = torch.tanh(new_c) * torch.sigmoid(o)

        return new_h, (new_c, new_h)


class haptic_feat(nn.Module):
    def __init__(self, HAPTIC_LAYER, ENC_LAYER, BEHAVIOR_LAYER):
        super(haptic_feat, self).__init__()
        self.HAPTIC_LAYER = HAPTIC_LAYER
        self.ENC_LAYER = ENC_LAYER
        self.BEHAVIOR_LAYER = BEHAVIOR_LAYER
        self.feature = nn.Sequential()
        self.feature.add_module("haptic_1", nn.Conv2d(1, 8, [5, 1], stride=[2, 1]))
        self.feature.add_module("relu_1", nn.ReLU())
        self.feature.add_module("hbn_1", nn.BatchNorm2d(8))
        self.feature.add_module("haptic_2", nn.Conv2d(8, 16, [3, 1], stride=[2, 1]))
        self.feature.add_module("relu_2", nn.ReLU())
        self.feature.add_module("hbn_2", nn.BatchNorm2d(16))
        self.feature.add_module("haptic_3", nn.Conv2d(16, self.HAPTIC_LAYER, [3, 3], stride=1))
        self.feature.add_module("relu_3", nn.ReLU())
        self.feature.add_module("hbn_3", nn.BatchNorm2d(self.HAPTIC_LAYER))
        self.feature_fusion = nn.Conv2d(self.HAPTIC_LAYER + self.ENC_LAYER + self.BEHAVIOR_LAYER, self.HAPTIC_LAYER,
                                        [3, 3], stride=1, padding=1)

        self.feature_lstm = ConvLSTM(in_channels=self.HAPTIC_LAYER, out_channels=self.HAPTIC_LAYER, kernel_size=5,
                                     padding=2)


    def forward(self, haptic, hap_specific, behavior_feat, state):
        haptic_feat = self.feature(haptic)
        hv_feat = torch.cat([haptic_feat, hap_specific, behavior_feat], dim=1)
        hv_feat = self.feature_fusion(hv_feat)
        hv_lstm_feat, hv_lstm_feat_state = self.feature_lstm(hv_feat, state)
        return hv_lstm_feat, haptic_feat, hv_lstm_feat_state


class audio_feat(nn.Module):
    def __init__(self, AUDIO_LAYER, ENC_LAYER, BEHAVIOR_LAYER):
        super(audio_feat, self).__init__()
        self.AUDIO_LAYER = AUDIO_LAYER
        self.ENC_LAYER = ENC_LAYER
        self.BEHAVIOR_LAYER = BEHAVIOR_LAYER
        self.feature = nn.Sequential()
        self.feature.add_module("audio_1", nn.Conv2d(1, 8, [1, 3], stride=[1, 2], padding=[0, 1]))
        self.feature.add_module("arelu_1", nn.ReLU())
        self.feature.add_module("abn_1", nn.BatchNorm2d(8))

        self.feature.add_module("audio_2", nn.Conv2d(8, 8, [1, 3], stride=[1, 2], padding=[0, 1]))
        self.feature.add_module("arelu_2", nn.ReLU())
        self.feature.add_module("abn_2", nn.BatchNorm2d(8))

        self.feature.add_module("audio_3", nn.Conv2d(8, 16, [1, 3], stride=[1, 2], padding=[0, 1]))
        self.feature.add_module("arelu_3", nn.ReLU())
        self.feature.add_module("abn_3", nn.BatchNorm2d(16))

        self.feature.add_module("audio_4", nn.Conv2d(16, self.AUDIO_LAYER, [3, 3], stride=[2, 2], padding=[1, 1]))
        self.feature.add_module("arelu_4", nn.ReLU())
        self.feature.add_module("abn_4", nn.BatchNorm2d(self.AUDIO_LAYER))
        self.feature_fusion = nn.Conv2d(self.AUDIO_LAYER + self.ENC_LAYER + self.BEHAVIOR_LAYER, self.AUDIO_LAYER,
                                        [3, 3], stride=1, padding=1)

        self.feature_lstm = ConvLSTM(in_channels=self.AUDIO_LAYER, out_channels=self.AUDIO_LAYER, kernel_size=5,
                                     padding=2)

    def forward(self, audio, aud_specific, behavior_feat, state):
        audio_feat = self.feature(audio)
        av_feat = torch.cat([audio_feat, aud_specific, behavior_feat], dim=1)
        av_feat = self.feature_fusion(av_feat)
        av_lstm_feat, av_lstm_feat_state = self.feature_lstm(av_feat, state)
        return av_lstm_feat, audio_feat, av_lstm_feat_state


class haptic_head(nn.Module):
    def __init__(self):
        super(haptic_head, self).__init__()
        self.head = nn.Sequential()
        self.head.add_module('deconv_1',
                             torch.nn.ConvTranspose2d(32, 16, [3, 3], stride=[1, 1]))
        self.head.add_module('relu_1', torch.nn.ReLU())
        self.head.add_module('deconv_2', torch.nn.ConvTranspose2d(16, 8, [3, 1], stride=[2, 1], output_padding=[1, 0]))
        self.head.add_module('relu_2', torch.nn.ReLU())
        self.head.add_module('deconv_3', torch.nn.ConvTranspose2d(8, 1, [5, 1], stride=[2, 1], output_padding=[1, 0]))
        self.head.add_module('relu_3', torch.nn.ReLU())

    def forward(self, haptic_feat, av_lstm_feat):
        feat = torch.cat([haptic_feat, av_lstm_feat], dim=1)
        haptic = self.head(feat)
        return haptic


class audio_head(nn.Module):
    def __init__(self):
        super(audio_head, self).__init__()
        self.head = nn.Sequential()
        self.head.add_module('deconv_1', torch.nn.ConvTranspose2d(32, 16, [3, 3], stride=[2, 2], output_padding=1, padding=1))
        self.head.add_module('relu_1', torch.nn.ReLU())
        self.head.add_module('deconv_2', torch.nn.ConvTranspose2d(16, 16, [1, 3], stride=[1, 2], output_padding=[0, 1], padding=[0, 1]))
        self.head.add_module('relu_2', torch.nn.ReLU())
        self.head.add_module('deconv_3', torch.nn.ConvTranspose2d(16, 8, [1, 3], stride=[1, 2], output_padding=[0, 1], padding=[0, 1]))
        self.head.add_module('relu_3', torch.nn.ReLU())
        self.head.add_module('deconv_4', torch.nn.ConvTranspose2d(8, 1, [1, 3], stride=[1, 2], output_padding=[0, 1], padding=[0, 1]))
        self.head.add_module('relu_4', torch.nn.ReLU())

    def forward(self, audio_feat, hv_lstm_feat):
        feat = torch.cat([audio_feat, hv_lstm_feat], dim=1)
        audio = self.head(feat)
        return audio
