import re
import os
import glob
import random
import numpy as np
import pickle
import pandas as pd
import PIL.Image

DATA_DIR = '../../data/CY101'
OUT_DIR = '../../data/CY101NPY'
IMG_WIDTH = 64
IMG_HEIGHT = 64


SEQUENCE_LENGTHS = {'crush': 49, 'grasp': 18, 'hold': 12, 'lift_slow': 43, 'look': 2, 'low_drop': 22,
                    'poke': 33, 'post_tap_look': 2, 'pre_tap_look': 2, 'push': 53, 'shake': 61, 'tap': 24 }

CATEGORIES = ['basket', 'weight', 'smallstuffedanimal', 'bigstuffedanimal', 'metal', 'timber', 'pasta', 'tin', 'pvc', 'cup',
              'can', 'bottle', 'cannedfood', 'medicine', 'tupperware', 'cone', 'noodle', 'eggcoloringcup', 'egg', 'ball']

CHOOSEN_BEHAVIORS = ['crush', 'poke', 'push']
SEQUENCE_LENGTH = 10
STEP = 4



def read_dir():
    visons = glob.glob(os.path.join(DATA_DIR, 'vision*/*/*/*/*'))
    haptics = glob.glob(os.path.join(DATA_DIR, 'rc_data/*/*/*/*/proprioception/ttrq0.txt'))
    audios = glob.glob(os.path.join(DATA_DIR, 'rc_data/*/*/*/*/hearing/*.wav'))
    vibros = glob.glob(os.path.join(DATA_DIR, 'rc_data/*/*/*/*/vibro/*.tsv'))
    return visons, haptics, audios, vibros


def generate_npy_vision(path):

    '''

    :param path: path to images folder,
    :return: numpy array with size [SUB_SAMPLE_SIZE, SEQ_LENGTH, ...]
    '''
    files = sorted(glob.glob(os.path.join(path, '*.jpg')))
    imglist = []
    for file in files:
        img = np.array(PIL.Image.open(file)).transpose([2, 0, 1])[np.newaxis, ...]
        imglist.append(img)
    ret = []
    for i in range(0, len(imglist)-SEQUENCE_LENGTH, STEP):
        ret.append(np.concatenate(imglist[i:i+SEQUENCE_LENGTH], axis=0))
    return ret, len(imglist)

def generate_npy_haptic(path, n_frames):
    '''
    :param path: path to ttrq0.txt, you need to open it before you process
    :return: list of numpy array with size [SEQ_LENGTH, ...]
    '''
    haplist = open(path, 'r').readlines()
    haplist = [list(map(float, v.strip().split('\t'))) for v in haplist]
    haplist = np.array(haplist)
    time_duration = (haplist[-1][0] - haplist[0][0])/n_frames
    bins = np.arange(haplist[0][0], haplist[-1][0], time_duration)
    groups = np.digitize(haplist[:,0], bins, right=False)

    haplist = [haplist[np.where(groups==idx)][...,1:] for idx in range(1, n_frames+1)]
    haplist = [np.append(ht, np.copy(ht[-1:, ...]), axis=0)[np.newaxis,...] if ht.shape[0] == 47 else ht[np.newaxis,...] for ht in haplist]
    ret = []
    for i in range(0, len(haplist) - SEQUENCE_LENGTH, STEP):
        ret.append(np.concatenate(haplist[i:i + SEQUENCE_LENGTH], axis=0))
    return ret


def generate_npy_audio(path, n_frames):
    '''
    :param path: path to ttrq0.txt, you need to open it before you process
    :return: list of numpy array with size [SEQ_LENGTH, ...]
    '''
    return [np.zeros([SEQUENCE_LENGTH, 7])]


def generate_npy_vibro(path):
    '''

    :param path: path to .tsv, you need to open it before you process
    :return: list of numpy array with size [SEQ_LENGTH, ...]
    '''
    return [np.zeros([SEQUENCE_LENGTH, 7])]


def process_vision(visions):
    train_subir = 'train'
    test_subir = 'test'
    if not os.path.exists(os.path.join(OUT_DIR, train_subir)):
        os.makedirs(os.path.join(OUT_DIR, train_subir))

    if not os.path.exists(os.path.join(OUT_DIR, test_subir)):
        os.makedirs(os.path.join(OUT_DIR, test_subir))

    random.shuffle(CATEGORIES)
    for vision in visions:
        save = False
        for bh in CHOOSEN_BEHAVIORS:
            save = save or (bh in vision)
        if save:
            subdir = ''
            for ct in CATEGORIES[:5]:
                if ct in vision:
                    subdir = test_subir
            for ct in CATEGORIES[5:]:
                if ct in vision:
                    subdir = train_subir

            out_sample_dir = os.path.join(OUT_DIR, subdir, '_'.join(vision.split('/')[-4:]))

            haptic = os.path.join(re.sub(r'vision_data_part[1-4]', 'rc_data', vision), 'proprioception', 'ttrq0.txt')
            audio = os.path.join(re.sub(r'vision_data_part[1-4]', 'rc_data', vision), 'hearing', '*.wav')
            vibro = os.path.join(re.sub(r'vision_data_part[1-4]', 'rc_data', vision), 'vibro', '*.tsv')
            out_vision_npys, n_frames = generate_npy_vision(vision)
            out_audio_npys = generate_npy_audio(audio, n_frames)
            out_haptic_npys = generate_npy_haptic(haptic, n_frames)
            out_vibro_npys = generate_npy_vibro(vibro)
            # make sure that all the lists are in the same length!
            for i, (out_vision_npy, out_haptic_npy, out_audio_npy, out_vibro_npy) in enumerate(zip(
                    out_vision_npys, out_haptic_npys, out_audio_npys, out_vibro_npys)):
                ret = {
                    'vision': out_vision_npy,
                    'haptic': out_haptic_npy,
                    'audio': out_audio_npy,
                    'vibro': out_vibro_npy
                }
                np.save(out_sample_dir+'_'+str(i), ret)


def run():

    visons, haptics, audios, vibros = read_dir()
    process_vision(visons)
    # process_haptic(haptics)



if __name__ == '__main__':
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        # for behavior in SEQUENCE_LENGTHS.keys():
        #     os.mkdir(os.path.join(OUT_DIR, behavior))
    run()