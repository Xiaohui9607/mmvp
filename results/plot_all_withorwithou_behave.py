import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

paths = {
    'PSNR':'Experiement_object_based_all_behavior_w_o_behave/psnr_all.csv',
    'SSIM': 'Experiement_object_based_all_behavior_w_o_behave/ssim_all.csv'
}
sns.set(style="whitegrid")
# sns.set(rc={'figure.figsize':(8.7,11.27)})

# f, axes = plt.subplots(1, 2, figsize=(16, 5))
# plt.rcParams.update({'errorbar.capsize': 3})
plt.rcParams.update({'font.size': 16})
plt.rc('ytick', labelsize=14)
plt.rc('xtick', labelsize=14)
f, axes = plt.subplots(1, 2, figsize=(13.5,4.5))
for i, metric in enumerate(paths):
    df = pd.read_csv(paths[metric])
    if i == len(paths)-1:
        axes[1].errorbar(x=range(4, 20), y=df.use_haptic_audio_vibro, yerr=df.use_haptic_audio_vibro_std,
                         marker="o", label="With Behavior Input Feature", fillstyle='none', color='orange')
        axes[1].errorbar(x=range(4, 20), y=df.use_haptic_audio_vibro_wo, yerr=df.use_haptic_audio_vibro_wo_std,
                         marker="x", label="Without Behavior Input Feature", fillstyle='none', color='green')
        axes[1].errorbar(x=range(4, 20), y=df.baseline, yerr=df.baseline_std, marker=8, label="Finn et al.",
                         fillstyle='none', color='blue')
        axes[i].set_ylim(0.65, 0.95)

    else:
        axes[0].errorbar(x=range(4, 20), y=df.use_haptic_audio_vibro, yerr=df.use_haptic_audio_vibro_std,
                         marker="o", fillstyle='none', color='orange')
        axes[0].errorbar(x=range(4, 20), y=df.use_haptic_audio_vibro_wo, yerr=df.use_haptic_audio_vibro_wo_std,
                         marker="x", fillstyle='none', color='green')
        axes[0].errorbar(x=range(4, 20),y=df.baseline, yerr=df.baseline_std,
                         marker=8, fillstyle='none', color='blue')
        axes[i].set_ylim(22, 32)

    axes[i].set_ylabel(metric, fontsize = 16)
    axes[i].set_xlabel("Time step", fontsize = 16)
    axes[i].set_title("Heldout set reconstruction evaluation", fontsize = 18)
    axes[i].xaxis.set_major_locator(ticker.MultipleLocator(2))
    if metric == 'SSIM':
        axes[i].yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    else:
        axes[i].yaxis.set_major_locator(ticker.MultipleLocator(2))

# plt.subplots_adjust(wspace=0.45)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('ablation_on_behavior.png', dpi=300)
plt.show()