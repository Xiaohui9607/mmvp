import os
import glob
import numpy as np
import PIL.Image

DATA_DIR = './data/CY101'
OUT_DIR = './data/CY101NPY'
IMG_WIDTH = 64
IMG_HEIGHT = 64

SEQUENCE_LENGTHS = {'crush': 49, 'grasp': 18, 'hold': 12, 'lift_slow': 43, 'look': 2, 'low_drop': 22,
                    'poke': 33, 'post_tap_look': 2, 'pre_tap_look': 2, 'push': 53, 'shake': 61, 'tap': 24 }

CHOOSEN_BEHAVIORS = ['crush', 'poke', 'push']
SEQUENCE_LENGTH = 10
STEP = 4


def read_dir():
    samples = glob.glob(os.path.join(DATA_DIR, '*/*/*/*/*'))
    return samples


def generate_npy(path):
    files = sorted(glob.glob(os.path.join(path, '*.jpg')))
    imglist = []
    for file in files:
        img = np.array(PIL.Image.open(file)).transpose([2, 0, 1])[np.newaxis, ...]
        imglist.append(img)
    ret = []
    for i in range(0, len(imglist)-SEQUENCE_LENGTH, STEP):
        ret.append(np.concatenate(imglist[i:i+SEQUENCE_LENGTH], axis=0))
    return ret


def run():
    samples = read_dir()

    for sample in samples:
        save = False
        for bh in CHOOSEN_BEHAVIORS:
            save = save or (bh in sample)
        if save:
            out_sample_dir = os.path.join(OUT_DIR, '_'.join(sample.split('/')[-4:]))

            out_sample_npys = generate_npy(sample)
            for i, subsample in enumerate(out_sample_npys):
                np.save(out_sample_dir+'_'+str(i), subsample)


if __name__ == '__main__':
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        # for behavior in SEQUENCE_LENGTHS.keys():
        #     os.mkdir(os.path.join(OUT_DIR, behavior))
    run()