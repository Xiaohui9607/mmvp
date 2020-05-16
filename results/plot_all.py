import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
paths = {
    'PSNR_OBJ': 'Experiement_object_based_all_behaviors/psnr_all.csv',
    'SSIM_OBJ': 'Experiement_object_based_all_behaviors/ssim_all.csv',
    'PSNR_CAT': 'Experiement_category_based_all_behavior/psnr_all.csv',
    'SSIM_CAT': 'Experiement_category_based_all_behavior/ssim_all.csv'
}
sns.set(style="darkgrid")
# sns.set(rc={'figure.figsize':(8.7,11.27)})

f, axes = plt.subplots(1, 4, figsize=(27,5))
for i, metric in enumerate(paths):
    df = pd.read_csv(paths[metric])
    if i == len(paths)-1:
        sns.lineplot(x=range(4, 20), y=df.baseline, ax=axes[i], label="Finn et al.")
        sns.lineplot(x=range(4, 20), y=df.use_haptic, ax=axes[i], label="vision+haptic")
        sns.lineplot(x=range(4, 20), y=df.use_haptic_audio, ax=axes[i], label="vision+haptic+audio")
        sns.lineplot(x=range(4, 20), y=df.use_haptic_audio_vibro, ax=axes[i], label="vision+haptic+audio+vibro")
    else:
        sns.lineplot(x=range(4, 20), y=df.baseline, ax=axes[i])
        sns.lineplot(x=range(4, 20), y=df.use_haptic, ax=axes[i])
        sns.lineplot(x=range(4, 20), y=df.use_haptic_audio, ax=axes[i])
        sns.lineplot(x=range(4, 20), y=df.use_haptic_audio_vibro, ax=axes[i])
    axes[i].set_ylabel(metric[:4], fontsize = 20)
    axes[i].set_xlabel("# frames", fontsize = 20)
    axes[i].set_title("Heldout set reconstruction evaluation", fontsize = 18)
    # plt.xlabel("# frames")

plt.legend(fontsize=12)
plt.savefig('all.png', dpi=600, bbox_inches='tight')
plt.show()