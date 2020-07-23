from options import Options
from model import Model
from metrics import mse_to_psnr
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

Experiement_object_based_sep_behaviors = {
    'crush': {
        'baseline': '/home/golf/code/models/Experiement_object_based_sep_behaviors/weight_baseline_crush/net_epoch_29.pth',
        'proposed': '/home/golf/code/models/Experiement_object_based_sep_behaviors/weight_use_haptic_audio_crush/net_epoch_29.pth',
    },
    # 'grasp': {
    #     'baseline': '/home/golf/code/models/Experiement_object_based_sep_behaviors/weight_baseline_grasp/net_epoch_29.pth',
    #     'proposed': '/home/golf/code/models/Experiement_object_based_sep_behaviors/weight_use_haptic_audio_grasp/net_epoch_29.pth',
    # },
    # 'hold': {
    #     'baseline': '/home/golf/code/models/Experiement_object_based_sep_behaviors/weight_baseline_hold/net_epoch_29.pth',
    #     'proposed': '/home/golf/code/models/Experiement_object_based_sep_behaviors/weight_use_haptic_audio_hold/net_epoch_29.pth',
    # },
    'lift_slow': {
        'baseline': '/home/golf/code/models/Experiement_object_based_sep_behaviors/weight_baseline_lift_slow/net_epoch_29.pth',
        'proposed': '/home/golf/code/models/Experiement_object_based_sep_behaviors/weight_use_haptic_audio_lift_slow/net_epoch_29.pth',
    },
    # 'low_drop': {
    #     'baseline': '/home/golf/code/models/Experiement_object_based_sep_behaviors/weight_baseline_low_drop/net_epoch_29.pth',
    #     'proposed': '/home/golf/code/models/Experiement_object_based_sep_behaviors/weight_use_haptic_audio_low_drop/net_epoch_29.pth',
    # },
    'poke': {
        'baseline': '/home/golf/code/models/Experiement_object_based_sep_behaviors/weight_baseline_poke/net_epoch_29.pth',
        'proposed': '/home/golf/code/models/Experiement_object_based_sep_behaviors/weight_use_haptic_audio_poke/net_epoch_29.pth',
    },
    'push': {
        'baseline': '/home/golf/code/models/Experiement_object_based_sep_behaviors/weight_baseline_push/net_epoch_29.pth',
        'proposed': '/home/golf/code/models/Experiement_object_based_sep_behaviors/weight_use_haptic_audio_push/net_epoch_29.pth',
    },
    'shake': {
        'baseline': '/home/golf/code/models/Experiement_object_based_sep_behaviors/weight_baseline_shake/net_epoch_29.pth',
        'proposed': '/home/golf/code/models/Experiement_object_based_sep_behaviors/weight_use_haptic_audio_shake/net_epoch_29.pth',
    },
    'tap': {
        'baseline': '/home/golf/code/models/Experiement_object_based_sep_behaviors/weight_baseline_tap/net_epoch_29.pth',
        'proposed': '/home/golf/code/models/Experiement_object_based_sep_behaviors/weight_use_haptic_audio_tap/net_epoch_29.pth',
    },
}


def eval_baseline(weight, behavior):
    opt = Options().parse()
    opt.baseline = True
    opt.sequence_length = 20
    opt.behavior_layer = 0
    opt.data_dir = '../data/'+behavior
    print("Model Config: ", opt)
    model = Model(opt)
    model.load_weight(weight)
    return model.evaluate(0, keep_frame=True)

def eval_proposed(weight, use_haptic, use_audio, use_virbo, behavior):
    opt = Options().parse()
    opt.use_behavior = True
    opt.use_haptic = use_haptic
    opt.use_audio = use_audio
    opt.use_vibro = False
    opt.behavior_layer = 1
    opt.aux = True
    opt.sequence_length = 20
    opt.data_dir = '../data/'+behavior
    print("Model Config: ", opt)
    model = Model(opt)
    model.load_weight(weight)
    return model.evaluate(0, keep_frame=True)

if __name__ == '__main__':
    f, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=False)
    sns.despine(left=True)

    frames = range(4, 20)
    for idx, setting in enumerate(Experiement_object_based_sep_behaviors):
        psnr_baseline = [mse_to_psnr(mse) for mse in eval_baseline(Experiement_object_based_sep_behaviors[setting]['baseline'],setting)]
        psnr_use_haptic_audio = [mse_to_psnr(mse) for mse in eval_proposed(Experiement_object_based_sep_behaviors[setting]['proposed'], True, True, False, setting)]
        if idx==0:
            sns.lineplot(x=range(4, 4+len(psnr_baseline)), y=psnr_baseline, legend='brief', label="baseline", ax=axes[idx % 3, idx // 3])
            sns.lineplot(x=range(4, 4+len(psnr_use_haptic_audio)), y=psnr_use_haptic_audio, legend='brief', label="+haptic+audio+vibro", ax=axes[idx % 3, idx // 3])
        else:
            sns.lineplot(x=range(4, 4+len(psnr_baseline)), y=psnr_baseline, legend='brief', ax=axes[idx % 3, idx // 3])
            sns.lineplot(x=range(4, 4+len(psnr_use_haptic_audio)), y=psnr_use_haptic_audio, legend='brief', ax=axes[idx % 3, idx // 3])
        axes[idx % 3, idx // 3].set_title(setting)
    axes[2, 1].set_xlabel("#frame")
    axes[2, 0].set_xlabel("#frame")
    axes[1, 0].set_ylabel("mse")
    plt.savefig("sep_ssim.png",dpi=600)
    plt.show()
