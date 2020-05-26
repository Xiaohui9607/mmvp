from options import Options
from model import Model
from metrics import  calc_ssim
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

Experiement_object_based_all_behaviors = {
    'baseline': '/home/golf/code/models/Experiement_object_based_all_behaviors/weight_baseline/net_epoch_29.pth',
    'weight_use_haptic': '/home/golf/code/models/Experiement_object_based_all_behaviors/weight_use_haptic/net_epoch_29.pth',
    'weight_use_haptic_audio': '/home/golf/code/models/Experiement_object_based_all_behaviors/weight_use_haptic_audio/net_epoch_29.pth',
    'weight_use_haptic_audio_vibro': '/home/golf/code/models/Experiement_object_based_all_behaviors/weight_use_haptic_audio_vibro/net_epoch_29.pth',
}

def eval_baseline(weight):
    opt = Options().parse()
    opt.baseline = True
    opt.sequence_length = 20
    print("Model Config: ", opt)
    model = Model(opt)
    model.load_weight(weight)
    return model.evaluate(0, keep_frame=True, ssim=False)

def eval_proposed(weight, use_haptic, use_audio, use_virbo):
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
    return model.evaluate(0, keep_frame=True, ssim=True)

if __name__ == '__main__':
    import pandas as pd

    data_dict = {}
    loss, std = eval_baseline(Experiement_object_based_all_behaviors['baseline'])
    metric_baseline = [(l, s) for l, s in zip(loss,std)]
    for frame_id, (loss, std) in enumerate(metric_baseline):
        data_dict[frame_id] = (loss, std)
    df = pd.DataFrame.from_dict(data_dict, orient="index")
    df.to_csv("{}.csv".format('baseline'))

    data_dict = {}
    loss, std = eval_proposed(Experiement_object_based_all_behaviors['weight_use_haptic'],
                              use_haptic=True, use_audio=False, use_virbo=False)
    metric_use_haptic = [(l, s) for l, s in zip(loss,std)]
    for frame_id, (loss, std) in enumerate(metric_use_haptic):
        data_dict[frame_id] = (loss, std)
    df = pd.DataFrame.from_dict(data_dict, orient="index")
    df.to_csv("{}.csv".format('use_haptic'))


    data_dict = {}
    loss, std =  eval_proposed(Experiement_object_based_all_behaviors['weight_use_haptic_audio'],
                               use_haptic=True, use_audio=True, use_virbo=False)
    metric_use_haptic_audio = [(l, s) for l, s in zip(loss,std)]
    for frame_id, (loss, std) in enumerate(metric_use_haptic_audio):
        data_dict[frame_id] = (loss, std)
    df = pd.DataFrame.from_dict(data_dict, orient="index")
    df.to_csv("{}.csv".format('use_haptic_audio'))

    data_dict = {}
    loss, std = eval_proposed(Experiement_object_based_all_behaviors['weight_use_haptic_audio_vibro'],
                              use_haptic=True, use_audio=True, use_virbo=True)
    metric_use_haptic_audio_vibro = [(l, s) for l, s in zip(loss,std)]
    for frame_id, (loss, std) in enumerate(metric_use_haptic_audio_vibro):
        data_dict[frame_id] = (loss, std)
    df = pd.DataFrame.from_dict(data_dict, orient="index")
    df.to_csv("{}.csv".format('use_haptic_audio_vibro'))


