import argparse
import os
import torch

# pylint: disable=C0103,C0301,R0903,W0622


class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--data_dir', default='../data/CY101NPY', help='directory containing data.')
        self.parser.add_argument('--channels', type=int, default=3, help='# channel of input')
        self.parser.add_argument('--height', type=int, default=64, help='height of image')
        self.parser.add_argument('--width', type=int, default=64, help='width of image')
        self.parser.add_argument('--output_dir', default='weight', help='directory for model weight.')
        self.parser.add_argument('--pretrained_model', default='', help='filepath of a pretrained model to initialize from.')
        self.parser.add_argument('--sequence_length', type=int, default=10, help='sequence length, including context frames.')
        self.parser.add_argument('--context_frames', type=int, default=4, help= '# of frames before predictions.')
        self.parser.add_argument('--use_haptic', action='store_true', help='Whether or not to give the haptic to the model')
        self.parser.add_argument('--use_behavior', action='store_true', help='Whether or not to give the behavior to the model')
        self.parser.add_argument('--use_audio', action='store_true', help='Whether or not to give the audio to the model')
        self.parser.add_argument('--model', default='CDNA', help='model architecture to use - CDNA, DNA, or STP')
        self.parser.add_argument('--num_masks', type=int, default=10, help='number of masks, usually 1 for DNA, 10 for CDNA, STN.')
        self.parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='cuda:[d] | cpu')
        self.parser.add_argument('--dna_kern_size', type=int, default=5, help='DNA_KERN_SIZE.')
        self.parser.add_argument('--haptic_layer', type=int, default=16, help='HAPTIC_LAYER.')
        self.parser.add_argument('--behavior_layer', type=int, default=9, help='number of chosen behavior.')
        self.parser.add_argument('--audio_layer', type=int, default=16, help='AUDIO_LAYER.')
        self.parser.add_argument('--baseline', action='store_true', help='baseline or proposed method, will override multi-modalities options' )
        self.parser.add_argument('--aux', action='store_true', help='employ auxiliary tasks during training' )
        # training details
        self.parser.add_argument('--print_interval', type=int, default=100, help='# iterations to output loss')
        self.parser.add_argument('--schedsamp_k', type=float, default=400, help='The k hyperparameter for scheduled sampling, -1 for no scheduled sampling.')
        self.parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        self.parser.add_argument('--learning_rate', type=float, default=0.001, help='the base learning rate of the generator')
        self.parser.add_argument('--epochs', type=int, default=30, help='# total training epoch')
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """
        self.opt = self.parser.parse_args()
        if not os.path.exists(self.opt.output_dir):
            os.makedirs(self.opt.output_dir)
        with open(os.path.join(self.opt.output_dir, "options"), 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(vars(self.opt).items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
        return self.opt

