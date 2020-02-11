import re
import os
import glob
import random
import numpy as np
import pickle
import pandas as pd
import PIL.Image
from make_spectrogram import plotstft

DATA_DIR = '../data/CY101'
OUT_DIR = '../data/CY101NPY'

max = [-10.478915, 1.017272, 6.426756, 5.950242, 0.75426, -0.013009, 0.034224]
min = [-39.090578, -21.720063, -10.159031, -4.562487, -1.456323, -1.893409, -0.080752]
mean = [-25.03760727, -8.2802204, -5.49065186, 2.53891808, -0.6424120, -1.22525292, -0.04463354]
std = [4.01142790e+01, 2.29780167e+01, 2.63156072e+01, 7.54091499e+00, 3.40810983e-01, 3.23891355e-01, 1.65208189e-03]

SEQUENCE_LENGTHS = {'crush': 49, 'grasp': 18, 'hold': 12, 'lift_slow': 43, 'look': 2, 'low_drop': 22,
                    'poke': 33, 'post_tap_look': 2, 'pre_tap_look': 2, 'push': 53, 'shake': 61, 'tap': 24 }


CATEGORIES = ['basket', 'weight', 'smallstuffedanimal', 'bigstuffedanimal', 'metal', 'timber', 'pasta', 'tin', 'pvc', 'cup',
              'can', 'bottle', 'cannedfood', 'medicine', 'tupperware', 'cone', 'noodle', 'eggcoloringcup', 'egg', 'ball']


CHOOSEN_BEHAVIORS = ['crush', 'poke', 'push']
SEQUENCE_LENGTH = 5
STEP = 4
IMG_SIZE = (64, 64)


def read_dir():
    visons = glob.glob(os.path.join(DATA_DIR, 'vision*/*/*/*/*'))
    return visons


def convert_audio_to_image(audio_path):
    ims = plotstft(audio_path)
    return ims


def generate_npy_vision(path):
    '''
    :param path: path to images folder,
    :return: numpy array with size [SUB_SAMPLE_SIZE, SEQ_LENGTH, ...]
    '''
    files = sorted(glob.glob(os.path.join(path, '*.jpg')))
    imglist = []
    for file in files:
        img = PIL.Image.open(file)
        img = img.resize(IMG_SIZE)
        img = np.array(img).transpose([2, 0, 1])[np.newaxis, ...]
        imglist.append(img)
    ret = []
    for i in range(0, len(imglist)-SEQUENCE_LENGTH, STEP):
        ret.append(np.concatenate(imglist[i:i+SEQUENCE_LENGTH], axis=0))
    return ret, len(imglist)


def generate_npy_haptic(path, n_frames):

    '''
    :param path: path to ttrq0.txt, you need to open it before you process
    :param n_frames: # frames
    :return: list of numpy array with size [SEQ_LENGTH, ...]
    :preprocess protocol: 48 bins for each single frame, given one frame, if #bin is less than 48,
                            we pad it in the tail with the last bin value. if #bin is more than 48, we take bin[:48]
    '''
    if not os.path.exists(path):
        return None
    haplist = open(path, 'r').readlines()
    haplist = [list(map(float, v.strip().split('\t'))) for v in haplist]
    haplist = np.array(haplist)
    time_duration = (haplist[-1][0] - haplist[0][0])/n_frames
    bins = np.arange(haplist[0][0], haplist[-1][0], time_duration)
    groups = np.digitize(haplist[:,0], bins, right=False)

    haplist = [haplist[np.where(groups==idx)][...,1:][:48] for idx in range(1, n_frames+1)]
    haplist = [np.pad(ht, [[0, 48-ht.shape[0]],[0, 0]] ,mode='edge')[np.newaxis,...] for ht in haplist]
    ret = []
    for i in range(0, len(haplist) - SEQUENCE_LENGTH, STEP):
        ret.append(np.concatenate(haplist[i:i + SEQUENCE_LENGTH], axis=0))
    return ret


def generate_npy_audio(path, n_frames_vision_image):
    '''
    :param path: path to audio, you need to open it before you process
    :return: list of numpy array with size [SEQ_LENGTH, ...]
    '''
    audio_path = glob.glob(path)
    if len(audio_path)==0:
        print(audio_path)
        return None
    audio_path = audio_path[0]
    converted_image_array = convert_audio_to_image(audio_path)

    #TODO delete these two lines
    # path = "../../data/spectrogram/cone_1/trial_1/exec_1/crush/hearing/cone_1_trial_1_exec_1_crush_hearing.png"
    # img = np.array(PIL.Image.open(path))[np.newaxis, np.newaxis, ...] # create a new dimension

    img = converted_image_array[np.newaxis, np.newaxis, ...] # create a new dimension
    image_width = img.shape[2]
    effective_each_frame_length = int(image_width/n_frames_vision_image)
    # here we need to crop from width
    width_to_keep = effective_each_frame_length * n_frames_vision_image
    cropped_image = img[:,:,:width_to_keep,:]
    imglist = []
    for i in range(0, n_frames_vision_image):
        imglist.append(cropped_image[:,:,i*effective_each_frame_length:(i+1)*effective_each_frame_length,:])

    ret = []
    for i in range(0, len(imglist) - SEQUENCE_LENGTH, STEP):
        ret.append(np.concatenate(imglist[i:i + SEQUENCE_LENGTH], axis=0))

    return ret


def generate_npy_vibro(path):
    '''

    :param path: path to .tsv, you need to open it before you process
    :return: list of numpy array with size [SEQ_LENGTH, ...]
    '''
    return [np.zeros([SEQUENCE_LENGTH, 7])]


def process(visions):
    train_subir = 'train'
    test_subir = 'test'
    if not os.path.exists(os.path.join(OUT_DIR, train_subir)):
        os.makedirs(os.path.join(OUT_DIR, train_subir))

    if not os.path.exists(os.path.join(OUT_DIR, test_subir)):
        os.makedirs(os.path.join(OUT_DIR, test_subir))

    random.shuffle(CATEGORIES)
    fail_count = 0
    for vision in visions:
        save = False
        for bh in CHOOSEN_BEHAVIORS:
            save = save or (bh in vision.split('/'))
        if not save:
            continue
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
        if out_audio_npys is None or out_haptic_npys is None:
            fail_count += 1
            continue

        print(len(out_haptic_npys), len(out_audio_npys), len(out_vision_npys))
        # out_vibro_npys = generate_npy_vibro(vibro)
        # make sure that all the lists are in the same length!
        for i, (out_vision_npy, out_haptic_npy, out_audio_npy) in enumerate(zip(
                out_vision_npys, out_haptic_npys, out_audio_npys)):
            ret = {
                'vision': out_vision_npy,
                'haptic': out_haptic_npy,
                'audio': out_audio_npy,
                # 'vibro': out_vibro_npy
            }
            np.save(out_sample_dir+'_'+str(i), ret)
    print(fail_count)

def run():

    visons = read_dir()
    process(visons)



if __name__ == '__main__':
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    run()