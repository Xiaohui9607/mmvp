import os
import glob
import numpy as np
import torch
import PIL.Image
from options import Options
from model import Model
import os, cv2
from torchvision.transforms import functional as F


Experiement_object_based_all_behaviors = {
    # 'baseline': '/home/golf/code/models/Experiement_object_based_all_behaviors/weight_baseline/net_epoch_29.pth',
    # 'weight_use_haptic': '/home/golf/code/models/Experiement_object_based_all_behaviors/weight_use_haptic/net_epoch_29.pth',
    # 'weight_use_haptic_audio': '/home/golf/code/models/Experiement_object_based_all_behaviors/weight_use_haptic_audio/net_epoch_29.pth',
    'weight_use_haptic_audio_vibro': '/home/golf/code/models/Experiement_object_based_all_behaviors/weight_use_haptic_audio_vibro/net_epoch_29.pth',
}

filelist = [
    # {'vision': '/home/golf/code/data/CY101/vision_data_part1/basket_funnel/trial_1/exec_1/push',
    #  'others': '/home/golf/code/data/CY101/rc_data/basket_funnel/trial_1/exec_1/push'},

    {'vision': '/home/golf/code/data/CY101/vision_data_part2/egg_rough_styrofoam/trial_1/exec_2/lift_slow',
     'others': '/home/golf/code/data/CY101/rc_data/egg_rough_styrofoam/trial_1/exec_2/lift_slow'},

    # {'vision': '/home/golf/code/data/CY101/vision_data_part4/timber_square/trial_1/exec_3/shake',
    #  'others': '/home/golf/code/data/CY101/rc_data/timber_square/trial_1/exec_3/shake'},
]


def predict_baseline(weight, filelist):

    opt = Options().parse()
    opt.baseline = True
    opt.sequence_length = 20
    print("Model Config: ", opt)
    model = Model(opt)
    model.load_weight(weight)
    resultlist, _ = model.predict(filelist)
    resultlist = [[(frame.squeeze().permute([1,2,0]).cpu().detach().numpy()*255).astype(np.uint8) for frame in result]
                  for result in resultlist]
    return resultlist


def predict_proposed(weight, use_haptic, use_audio, use_virbo, filelist):
    opt = Options().parse()
    opt.use_behavior = True
    opt.use_haptic = use_haptic
    opt.use_audio = use_audio
    opt.use_vibro = use_virbo
    opt.aux = True
    opt.sequence_length = 20
    print("Model Config: ", opt)
    model = Model(opt)
    model.load_weight(weight)
    resultlist, gt = model.predict(filelist)
    gen_audios = [np.vstack([hp.cpu().numpy().squeeze()[-3:] for hp in haptic]) for _, haptic, _, _ in resultlist]
    gt_audios = [np.vstack([hp.cpu().numpy().squeeze()[:,-3:] for hp in haptic]) for _, haptic, _, _  in gt]
    pass


if __name__ == '__main__':
    # resultlist = predict_baseline(Experiement_object_based_all_behaviors['baseline'], filelist)
    # for idx, result in enumerate(resultlist):
    #     os.mkdir("baseline_{}".format(idx))
    #     for id, frame in enumerate(result):
    #         cv2.imwrite("baseline_{}/{}.png".format(idx, id), cv2.cvtColor(cv2.resize(frame,(128,128)), cv2.COLOR_BGR2RGB))

    resultlist, gtlist = predict_proposed(Experiement_object_based_all_behaviors['weight_use_haptic_audio_vibro'], True, True, True, filelist)
    # for idx, result in enumerate(resultlist):
    #     os.mkdir("haptic_{}".format(idx))
    #     for id, frame in enumerate(result):
    #         cv2.imwrite("haptic_{}/{}.png".format(idx, id), cv2.cvtColor(cv2.resize(frame,(128,128)), cv2.COLOR_BGR2RGB))


    # for idx, result in enumerate(gtlist):
    #     os.mkdir("groundtruth_{}".format(idx))
    #     for id, frame in enumerate(result):
    #         cv2.imwrite("groundtruth_{}/{}.png".format(idx, id), cv2.cvtColor(cv2.resize(frame,(128,128)), cv2.COLOR_BGR2RGB))






