from options import Options
from model import Model
from metrics import mse_to_psnr, calc_ssim
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
    return model.evaluate(0, keep_frame=True)

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
    return model.evaluate(0, keep_frame=True)

if __name__ == '__main__':
    psnr_baseline = [mse_to_psnr(mse) for mse in eval_baseline(Experiement_object_based_all_behaviors['baseline'])]
    psnr_use_haptic= \
        [mse_to_psnr(mse)for mse in eval_proposed(Experiement_object_based_all_behaviors['weight_use_haptic'],
                                                                use_haptic=True, use_audio=False, use_virbo=False)]
    psnr_use_haptic_audio = \
        [mse_to_psnr(mse) for mse in eval_proposed(Experiement_object_based_all_behaviors['weight_use_haptic_audio'],
                                                                       use_haptic=True, use_audio=True, use_virbo=False)]
    psnr_use_haptic_audio_vibro = \
        [mse_to_psnr(mse) for mse in eval_proposed(Experiement_object_based_all_behaviors['weight_use_haptic_audio_vibro'],
                                                                             use_haptic=True, use_audio=True, use_virbo=True)]

    frames = range(4, 20)

    sns.lineplot(x=frames, y=psnr_baseline, legend='brief', label="baseline")
    sns.lineplot(x=frames, y=psnr_use_haptic, legend='brief', label="+haptic+audio+vibro")
    sns.lineplot(x=frames, y=psnr_use_haptic_audio, legend='brief', label="+haptic+audio")
    sns.lineplot(x=frames, y=psnr_use_haptic_audio_vibro, legend='brief', label="+haptic")
    plt.xlabel('# frame')
    plt.ylabel('SSIM')
    plt.savefig("all_ssim.png",dpi=300)
    plt.show()
