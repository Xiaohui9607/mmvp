import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker


paths = {
    'OBJ':{
        'PSNR': 'Experiement_object_based_all_behaviors/psnr_all.csv',
        'SSIM': 'Experiement_object_based_all_behaviors/ssim_all.csv',
    },
    'CAT':{
        'PSNR': 'Experiement_category_based_all_behavior/psnr_all.csv',
        'SSIM': 'Experiement_category_based_all_behavior/ssim_all.csv'
    }
}

# behaviors = ['crush', 'lift', 'grasp', 'shake', 'push', 'tap']
settings = ['baseline', 'use_haptic', 'use_haptic_audio', 'use_haptic_audio_vibro']
sns.set(style="darkgrid")
lengends = {
    'baseline':'vision (Finn et al.)',
    'use_haptic':'vision+haptic',
    'use_haptic_audio': 'vision+haptic+audio' ,
    'use_haptic_audio_vibro': 'vision+haptic+audio+vibro'
}
# f, axes = plt.subplots(1, 2, figsize=(19,5))
fig = plt.figure(figsize=(29,5))
outer = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.2)

for i, TYPE in enumerate(paths):
    axes = gridspec.GridSpecFromSubplotSpec(1, 2,
                                             subplot_spec=outer[i], wspace=0.2, hspace=0.5)
    for j, metric in enumerate(paths[TYPE]):
        df = pd.read_csv(paths[TYPE][metric])
        ax = plt.Subplot(fig, axes[j])
        for setting in settings:
            ys = df[setting]
            if i*j == 1:
                if lengends[setting]=='vision (Finn et al.)':
                    sns.lineplot(x=range(4, 20), y=ys, ax=ax,  marker=8, label=lengends[setting])
                else:
                    sns.lineplot(x=range(4, 20), y=ys, ax=ax, marker="o", label=lengends[setting])
            else:
                if lengends[setting] == 'vision (Finn et al.)':
                    sns.lineplot(x=range(4, 20), y=ys, ax=ax, marker=8)
                else:
                    sns.lineplot(x=range(4, 20), y=ys, ax=ax, marker="o")
        ax.set_xlabel("Time step", fontsize=16)
        ax.set_ylabel(metric, fontsize=16)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

        fig.add_subplot(ax)

plt.legend(fontsize=16, loc='center', bbox_to_anchor=(-1.35, -0.2, 0, 0),
           ncol=4, columnspacing=15,frameon=False)
fig.suptitle('model performance on object-based dataset'
             '                                 '
             '                                 '
             'model performance on category-based dataset', fontsize=20)
plt.savefig('all.png', dpi=300, bbox_inches='tight')
fig.show()