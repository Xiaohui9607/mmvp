from options import Options
from model import Model
from metrics import mse_to_psnr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

Experiement_object_based_sep_behaviors = [
    '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_baseline_crush/net_epoch_29.pth',
    '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_baseline_grasp/net_epoch_29.pth',
    '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_baseline_hold/net_epoch_29.pth',
    '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_baseline_lift_slow/net_epoch_29.pth',
    '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_baseline_low_drop/net_epoch_29.pth',
    '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_baseline_poke/net_epoch_29.pth',
    '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_baseline_push/net_epoch_29.pth',
    '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_baseline_shake/net_epoch_29.pth',
    '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_baseline_tap/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_audio_crush/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_audio_grasp/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_audio_hold/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_audio_lift_slow/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_audio_low_drop/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_audio_poke/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_audio_push/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_audio_shake/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_audio_tap/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_audio_vibro_crush/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_audio_vibro_grasp/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_audio_vibro_hold/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_audio_vibro_lift_slow/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_audio_vibro_low_drop/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_audio_vibro_poke/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_audio_vibro_push/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_audio_vibro_shake/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_audio_vibro_tap/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_crush/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_grasp/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_hold/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_lift_slow/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_low_drop/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_poke/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_push/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_shake/net_epoch_29.pth',
    # '/home/golf/code/models/Experiement_object_based_sep_behaviors_lighter_network/weight_use_haptic_tap/net_epoch_29.pth',
]


def eval_baseline(weight, behavior):
    opt = Options().parse()
    opt.baseline = True
    opt.sequence_length = 20
    opt.behavior_layer = 0
    opt.data_dir = '../data/'+behavior
    print("Model Config: ", opt)
    model = Model(opt)
    model.load_weight(weight)
    return model.evaluate(0, keep_batch=True, ssim=True)

def eval_proposed(weight, use_haptic, use_audio, use_virbo, behavior):
    opt = Options().parse()
    opt.use_behavior = True
    opt.use_haptic = use_haptic
    opt.use_audio = use_audio
    opt.use_vibro = use_virbo
    opt.behavior_layer = 1
    opt.aux = True
    opt.sequence_length = 20
    opt.data_dir = '../data/'+behavior
    print("Model Config: ", opt)
    model = Model(opt)
    model.load_weight(weight)
    return model.evaluate(0, keep_batch=True, ssim=True)


if __name__ == '__main__':
    csv_columns = ['setting', 'SSIM']
    for idx, path in enumerate(Experiement_object_based_sep_behaviors):
        # psnr_baseline = [mse_to_psnr(mse) for mse in eval_baseline(Experiement_object_based_sep_behaviors[setting]['baseline'],setting)]
        behavior = path.split('/')[-2].split('_')[-1]
        out = path.split('/')[-2]
        use_haptic = 'haptic' in path
        use_audio = 'audio' in path
        use_virbo = 'vibro' in path
        if behavior == 'slow':
            behavior = 'lift_slow'
        if behavior == 'drop':
            behavior = 'low_drop'
        pass
        dict_data = {}
        rowid = 0
        psnr_baseline = eval_baseline(Experiement_object_based_sep_behaviors[idx], behavior)

        # psnr_use_haptic_audio = eval_proposed(path, use_haptic, use_audio, use_virbo, behavior)
        for ssim in psnr_baseline:
            dict_data[rowid] = ('{}_{}_{}_{}'.format(behavior, use_haptic, use_audio, use_virbo), ssim)
            rowid+=1
        df = pd.DataFrame.from_dict(dict_data,orient="index")
        df.to_csv("{}.csv".format(out))
