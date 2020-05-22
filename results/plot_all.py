import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

paths = {
    'PSNR':'Experiement_object_based_all_behaviors/psnr_all.csv',
    'SSIM': 'Experiement_object_based_all_behaviors/ssim_all.csv'
}

sns.set(style="darkgrid")
# sns.set(rc={'figure.figsize':(8.7,11.27)})

# f, axes = plt.subplots(1, 2, figsize=(16, 5))

plt.rcParams.update({'font.size': 20})
plt.rc('ytick', labelsize=14)
plt.rc('xtick', labelsize=14)
f, axes = plt.subplots(1, 2, figsize=(13.5,4.5))
for i, metric in enumerate(paths):
    df = pd.read_csv(paths[metric])
    if i == len(paths)-1:
        sns.lineplot(x=range(4, 20), y=df.use_haptic_audio_vibro, ax=axes[i], marker="o", label="vision+haptic+audio+vibro", color='red')
        sns.lineplot(x=range(4, 20), y=df.use_haptic_audio, ax=axes[i], marker="o", label="vision+haptic+audio", color='green')
        sns.lineplot(x=range(4, 20), y=df.use_haptic, ax=axes[i], marker="o", label="vision+haptic", color='orange')
        sns.lineplot(x=range(4, 20), y=df.baseline, ax=axes[i], marker=8, label="Finn et al.", color='blue')
    else:
        sns.lineplot(x=range(4, 20), y=df.use_haptic_audio_vibro, ax=axes[i], marker="o", color='red')#, label="vision+haptic+audio+vibro")
        sns.lineplot(x=range(4, 20), y=df.use_haptic_audio, ax=axes[i], marker="o", color='green')#, label="vision+haptic+audio")
        sns.lineplot(x=range(4, 20), y=df.use_haptic, ax=axes[i], marker="o", color='orange')  # , label="vision+haptic")
        sns.lineplot(x=range(4, 20), y=df.baseline, ax=axes[i], marker=8, color='blue')  # , label="Finn et al.")
    axes[i].set_ylabel(metric, fontsize = 20)
    axes[i].set_xlabel("Time step", fontsize = 20)
    axes[i].set_title("Heldout set reconstruction evaluation", fontsize = 20)
    # plt.xlabel("# frames")
    axes[i].xaxis.set_major_locator(ticker.MultipleLocator(2))
    if metric == 'SSIM':
        axes[i].yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    else:
        axes[i].yaxis.set_major_locator(ticker.MultipleLocator(1))

# plt.subplots_adjust(wspace=0.45)
plt.legend(fontsize=12)
plt.savefig('all_behave.png', dpi=300, bbox_inches='tight')
plt.show()