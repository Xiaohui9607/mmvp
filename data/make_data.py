import re
import os
import glob
import random
import numpy as np
import PIL.Image
from make_spectrogram import plotstft
import argparse
from sklearn.preprocessing import OneHotEncoder


DATA_DIR = '../data/CY101'
OUT_DIR = '../data/CY101NPY'


CATEGORIES = ['basket', 'weight', 'smallstuffedanimal', 'bigstuffedanimal', 'metal', 'timber', 'pasta', 'tin', 'pvc',
              'cup', 'can', 'bottle', 'cannedfood', 'medicine', 'tupperware', 'cone', 'noodle', 'eggcoloringcup', 'egg',
              'ball']
OBJECTS = [
    'ball_base', 'can_coke', 'egg_rough_styrofoam', 'noodle_3', 'timber_square', 'ball_basket', 'can_red_bull_large',
    'egg_smooth_styrofoam', 'noodle_4', 'timber_squiggle', 'ball_blue', 'can_red_bull_small', 'egg_wood', 'noodle_5',
    'tin_pokemon',
    'ball_transparent', 'can_starbucks', 'eggcoloringcup_blue', 'pasta_cremette', 'tin_poker', 'ball_yellow_purple',
    'cannedfood_chili',
    'eggcoloringcup_green', 'pasta_macaroni', 'tin_snack_depot', 'basket_cylinder', 'cannedfood_cowboy_cookout',
    'eggcoloringcup_orange',
    'pasta_penne', 'tin_snowman', 'basket_funnel', 'cannedfood_soup', 'eggcoloringcup_pink', 'pasta_pipette', 'tin_tea',
    'basket_green',
    'cannedfood_tomato_paste', 'eggcoloringcup_yellow', 'pasta_rotini', 'tupperware_coffee_beans', 'basket_handle',
    'cannedfood_tomatoes',
    'medicine_ampicillin', 'pvc_1', 'tupperware_ground_coffee', 'basket_semicircle', 'cone_1', 'medicine_aspirin',
    'pvc_2', 'tupperware_marbles',
    'bigstuffedanimal_bear', 'cone_2', 'medicine_bilberry_extract', 'pvc_3', 'tupperware_pasta',
    'bigstuffedanimal_bunny', 'cone_3',
    'medicine_calcium', 'pvc_4', 'tupperware_rice', 'bigstuffedanimal_frog', 'cone_4', 'medicine_flaxseed_oil', 'pvc_5',
    'weight_1',
    'bigstuffedanimal_pink_dog', 'cone_5', 'metal_flower_cylinder', 'smallstuffedanimal_bunny', 'weight_2',
    'bigstuffedanimal_tan_dog',
    'cup_blue', 'metal_food_can', 'smallstuffedanimal_chick', 'weight_3', 'bottle_fuse', 'cup_isu',
    'metal_mix_covered_cup',
    'smallstuffedanimal_headband_bear', 'weight_4', 'bottle_google', 'cup_metal', 'metal_tea_jar',
    'smallstuffedanimal_moose',
    'weight_5', 'bottle_green', 'cup_paper_green', 'metal_thermos', 'smallstuffedanimal_otter', 'bottle_red',
    'cup_yellow', 'timber_pentagon', 'bottle_sobe', 'egg_cardboard', 'noodle_1', 'timber_rectangle', 'can_arizona',
    'egg_plastic_wrap', 'noodle_2', 'timber_semicircle'
]


# Objects
SORTED_OBJECTS = sorted(OBJECTS)


#CHOOSEN_BEHAVIORS = ['crush', 'grasp', 'lift_slow', 'shake', 'poke', 'push']
CHOOSEN_BEHAVIORS = []

crop_stategy = {
    'crush': [16, -5],
    'grasp': [0, -13],
    'lift_slow': [0, -3],
    'shake': [0, -1],
    'poke': [2, -16],
    'push': [2, -13]
}

SEQUENCE_LENGTH = 10
STEP = 4
IMG_SIZE = (64, 64)


def read_dir():
    visions = glob.glob(os.path.join(DATA_DIR, 'vision*/*/*/*/*'))
    return visions


def convert_audio_to_image(audio_path):
    ims = plotstft(audio_path)
    return ims


def generate_npy_vision(path, behavior):
    '''
    :param path: path to images folder,
    :return: numpy array with size [SUB_SAMPLE_SIZE, SEQ_LENGTH, ...]
    '''
    files = sorted(glob.glob(os.path.join(path, '*.jpg')))
    img_length = len(files)
    files = files[crop_stategy[behavior][0]:crop_stategy[behavior][1]]
    imglist = []
    for file in files:
        img = PIL.Image.open(file)
        img = img.resize(IMG_SIZE)
        img = np.array(img).transpose([2, 0, 1])[np.newaxis, ...]
        imglist.append(img)
    ret = []
    for i in range(0, len(imglist) - SEQUENCE_LENGTH, STEP):
        ret.append(np.concatenate(imglist[i:i + SEQUENCE_LENGTH], axis=0))
    return ret, img_length


def generate_npy_haptic(path1, path2, n_frames, behavior):
    '''
    :param path: path to ttrq0.txt, you need to open it before you process
    :param n_frames: # frames
    :return: list of numpy array with size [SEQ_LENGTH, ...]
    :preprocess protocol: 48 bins for each single frame, given one frame, if #bin is less than 48,
                            we pad it in the tail with the last bin value. if #bin is more than 48, we take bin[:48]
    '''
    if not os.path.exists(path1):
        return None
    haplist1 = open(path1, 'r').readlines()
    haplist2 = open(path2, 'r').readlines()
    haplist = [list(map(float, v.strip().split('\t'))) + list(map(float, w.strip().split('\t')))[1:] for v, w in
               zip(haplist1, haplist2)]
    haplist = np.array(haplist)
    time_duration = (haplist[-1][0] - haplist[0][0]) / n_frames
    bins = np.arange(haplist[0][0], haplist[-1][0], time_duration)
    groups = np.digitize(haplist[:, 0], bins, right=False)

    haplist = [haplist[np.where(groups == idx)][..., 1:][:48] for idx in range(1, n_frames + 1)]
    haplist = [np.pad(ht, [[0, 48 - ht.shape[0]], [0, 0]], mode='edge')[np.newaxis, ...] for ht in haplist]
    haplist = haplist[crop_stategy[behavior][0]:crop_stategy[behavior][1]]
    ret = []
    for i in range(0, len(haplist) - SEQUENCE_LENGTH, STEP):
        ret.append(np.concatenate(haplist[i:i + SEQUENCE_LENGTH], axis=0).astype(np.float32)[:, np.newaxis, ...])
    return ret


def generate_npy_audio(path, n_frames_vision_image, behavior):
    '''
    :param path: path to audio, you need to open it before you process
    :return: list of numpy array with size [SEQ_LENGTH, ...]
    '''
    audio_path = glob.glob(path)
    if len(audio_path) == 0:
        return None
    audio_path = audio_path[0]
    converted_image_array = convert_audio_to_image(audio_path)

    img = converted_image_array[np.newaxis, ...]  # create a new dimension

    image_width = img.shape[1]
    effective_each_frame_length = int(image_width / n_frames_vision_image)
    # here we need to crop from width
    width_to_keep = effective_each_frame_length * n_frames_vision_image
    cropped_image = img[:, :width_to_keep, :]
    imglist = []
    for i in range(0, n_frames_vision_image):
        imglist.append(cropped_image[:, i * effective_each_frame_length:(i + 1) * effective_each_frame_length, :])
    imglist = imglist[crop_stategy[behavior][0]:crop_stategy[behavior][1]]
    ret = []
    for i in range(0, len(imglist) - SEQUENCE_LENGTH, STEP):
        ret.append(np.concatenate(imglist[i:i + SEQUENCE_LENGTH], axis=0).astype(np.float32)[:, np.newaxis, ...])
    return ret


def generate_npy_vibro(path):
    '''
    :param path: path to .tsv, you need to open it before you process
    :return: list of numpy array with size [SEQ_LENGTH, ...]
    '''
    return [np.zeros([SEQUENCE_LENGTH, 7])]


def process(visions, chosen_behavior):
    CHOOSEN_BEHAVIORS.append(chosen_behavior)
    train_subir = 'train'
    test_subir = 'test'
    if not os.path.exists(os.path.join(OUT_DIR, train_subir)):
        os.makedirs(os.path.join(OUT_DIR, train_subir))

    if not os.path.exists(os.path.join(OUT_DIR, test_subir)):
        os.makedirs(os.path.join(OUT_DIR, test_subir))

    train_list = []
    test_list = []
    for i in range(len(SORTED_OBJECTS)//5):
        random_number = np.random.randint(low=0, high=5)
        np.random.shuffle(SORTED_OBJECTS[5*i:5*i+5])
        shuffled_category = SORTED_OBJECTS[5*i:5*i+5]
        for item, object in enumerate(shuffled_category):
            if item == random_number:
                test_list.append(object)
            else:
                train_list.append(object)

    split_base = train_list + test_list
    cutting = len(train_list)

    fail_count = 0
    for vision in visions:
        save = False
        behavior = ''
        for _bh in CHOOSEN_BEHAVIORS:
            if _bh in vision.split('/'):
                behavior = _bh
                save = True
                break
        if not save:
            continue
        subdir = ''
        for ct in split_base[:cutting]:
            if ct in vision:
                subdir = train_subir
        for ct in split_base[cutting:]:
            if ct in vision:
                subdir = test_subir

        out_sample_dir = os.path.join(OUT_DIR, subdir, '_'.join(vision.split('/')[-4:]))
        # behavior = out_sample_dir.split('_')[-1]
        # print(behavior)

        haptic1 = os.path.join(re.sub(r'vision_data_part[1-4]', 'rc_data', vision), 'proprioception', 'ttrq0.txt')
        haptic2 = os.path.join(re.sub(r'vision_data_part[1-4]', 'rc_data', vision), 'proprioception', 'cpos0.txt')
        audio = os.path.join(re.sub(r'vision_data_part[1-4]', 'rc_data', vision), 'hearing', '*.wav')
        vibro = os.path.join(re.sub(r'vision_data_part[1-4]', 'rc_data', vision), 'vibro', '*.tsv')
        out_vision_npys, n_frames = generate_npy_vision(vision, behavior)
        out_audio_npys = generate_npy_audio(audio, n_frames, behavior)
        out_haptic_npys = generate_npy_haptic(haptic1, haptic2, n_frames, behavior)
        if out_audio_npys is None or out_haptic_npys is None:
            fail_count += 1
            continue
        out_behavior_npys = np.zeros(len(CHOOSEN_BEHAVIORS))
        out_behavior_npys[CHOOSEN_BEHAVIORS.index(behavior)] = 1

        # print(len(out_haptic_npys), len(out_audio_npys), len(out_vision_npys))
        # out_vibro_npys = generate_npy_vibro(vibro)
        # make sure that all the lists are in the same length!
        for i, (out_vision_npy, out_haptic_npy, out_audio_npy) in enumerate(zip(
                out_vision_npys, out_haptic_npys, out_audio_npys)):
            ret = {
                'behavior': out_behavior_npys,
                'vision': out_vision_npy,
                'haptic': out_haptic_npy,
                'audio': out_audio_npy,
            }
            np.save(out_sample_dir + '_' + str(i), ret)
    print("fail: ", fail_count)


def run(chosen_behavior):
    print("start making data")
    visons = read_dir()
    process(visons, chosen_behavior)
    print("done!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--behavior', default='crush', help='which behavior?')
    args = parser.parse_args()

    print("behavior: ", args.behavior)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    run(chosen_behavior=args.behavior)