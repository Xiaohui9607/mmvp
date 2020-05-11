import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker


paths = {
    'PSNR': 'Experiement_object_based_sep_behaviors_lighter_network/psnr_sep.csv',
    'SSIM': 'Experiement_object_based_sep_behaviors_lighter_network/ssim_sep.csv'
}
behaviors = ['crush', 'lift', 'grasp', 'shake', 'push', 'tap']
settings = ['baseline', 'haptic', 'haptic_audio', 'haptic_audio_vibro']
sns.set(style="darkgrid")
lengends = {
    'baseline':'Finn et al.',
    'haptic':'vision+haptic',
    'haptic_audio': 'vision+haptic+audio' ,
    'haptic_audio_vibro': 'vision+haptic+audio+vibro'
}
# f, axes = plt.subplots(1, 2, figsize=(19,5))
fig = plt.figure(figsize=(31,8))
outer = gridspec.GridSpec(1, 2, wspace=0.15, hspace=0.2)
for i, metric in enumerate(paths):
    df = pd.read_csv(paths[metric])
    # ax1 =
    axes = gridspec.GridSpecFromSubplotSpec(2, 3,
                                             subplot_spec=outer[i], wspace=0.15, hspace=0.5)
    for j, behave in enumerate(behaviors):
        ax = plt.Subplot(fig, axes[j // 3, j % 3])
        for setting in settings:
            ys = df['{}_{}'.format(setting, behave)]
            xs = range(4, 10) if behave in ['grasp', 'tap'] else range(4, 20)
            ys = ys[:len(list(xs))]
            if i*j == len(behaviors)-1:
                sns.lineplot(x=xs, y=ys, ax=ax, label=lengends[setting])
            else:
                sns.lineplot(x=xs, y=ys, ax=ax)
        if j in [0, 3]:
            ax.set_ylabel(metric)
        else:
            ax.yaxis.label.set_visible(False)
        ax.set_xlabel("# frames")
        ax.set_title(behave, fontsize=14)
        if metric == 'SSIM':
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        else:
            if behave in ['grasp', 'hold', 'tap']:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            else:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

        fig.add_subplot(ax)

# plt.legend(ncol=4)
plt.legend(fontsize=16, loc='center', bbox_to_anchor=(-3, -0.55, 0.5, 0.5),
           ncol=4, columnspacing=15,frameon=False)
fig.suptitle('Heldout set reconstruction evaluation on separated behaviours', fontsize=20)
plt.savefig('sep.png', dpi=600, bbox_inches='tight')
fig.show()