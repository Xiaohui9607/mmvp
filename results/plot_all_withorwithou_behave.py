import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
paths = {
    'PSNR': 'Experiement_object_based_all_behavior_w_o_behave/psnr_all.csv',
    'SSIM': 'Experiement_object_based_all_behavior_w_o_behave/ssim_all.csv',
}
sns.set(style="darkgrid")
# sns.set(rc={'figure.figsize':(8.7,11.27)})

f, axes = plt.subplots(1, 2, figsize=(19,5))
for i, metric in enumerate(paths):
    df = pd.read_csv(paths[metric])
    if i == len(paths)-1:
        sns.lineplot(x=range(4, 20), y=df.baseline, ax=axes[i], label="Finn et al.")
        sns.lineplot(x=range(4, 20), y=df.use_haptic_audio_vibro, ax=axes[i], label="with behave")
        sns.lineplot(x=range(4, 20), y=df.use_haptic_audio_vibro_wo, ax=axes[i], label="without behave")
    else:
        sns.lineplot(x=range(4, 20), y=df.baseline, ax=axes[i])
        sns.lineplot(x=range(4, 20), y=df.use_haptic_audio_vibro, ax=axes[i])
        sns.lineplot(x=range(4, 20), y=df.use_haptic_audio_vibro_wo, ax=axes[i])
    axes[i].set_ylabel(metric, fontsize = 20)
    axes[i].set_xlabel("# frames", fontsize = 20)
    axes[i].set_title("Heldout set reconstruction evaluation", fontsize = 18)
    # plt.xlabel("# frames")

plt.legend(fontsize=12)
plt.savefig('all_wo.png', dpi=600, bbox_inches='tight')
plt.show()