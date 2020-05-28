import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
sns.set(style="whitegrid")

path = './sepvsall'

dpi = 300
plt.rc('ytick', labelsize=12)
plt.rc('xtick', labelsize=12)

fig = plt.figure(figsize=(6100/300.0, 1800/300.0))
behaviors = ['crush', 'lift', 'grasp', 'shake', 'push', 'tap', 'hold', 'drop', 'poke', 'all']
settings = ['baseline', 'haptic', 'haptic_audio', 'haptic_audio_vibro']
plist = []

for i, behavior in enumerate(behaviors):
    xs = []
    ys = []
    dfs = []
    for j, a_setting in enumerate(settings):
        df = pd.read_csv(os.path.join(path, a_setting, '{}.csv'.format(behavior)))
        dfs.append(df.iloc[:, 1])
        # add ensemble
        # add ensemble

    p = pd.DataFrame(dfs).T
    p.columns=settings
    p1 = p.melt(var_name='setting', value_name='SSIM')
    p1 = p1.dropna(axis=0,how='any')
    p1['behavior']= behavior
    plist.append(p1)

# ensemble

for i, setting in enumerate(settings):
    xs = []
    ys = []
    dfs = []
    for j, behave in enumerate(behaviors[:-1]):
        df = pd.read_csv(os.path.join(path, setting, '{}.csv'.format(behave)))
        dfs.append(df.iloc[:, 1])
    # ax = axes[i]
    p = pd.DataFrame(dfs)
    p = p.mean().T
    p = pd.DataFrame(p)
    p.columns = [setting]
    p1 = p.melt(var_name='setting', value_name='SSIM')
    p1['behavior']= 'ensemble'
    plist.append(p1)


p = pd.concat(plist, axis=0)
g = sns.catplot(x="behavior", y="SSIM", hue="setting", data=p, kind="bar",
                col_order=['blue', 'orange', 'green', 'red'], aspect=3, legend=False, ci='sd')
g.despine(left=True)

plt.ylim([0.7, 1.0])
plt.xlabel('behavior', fontsize=16)
plt.ylabel('SSIM', fontsize=16)

# reodering the legends and renaming them
handles, _ = plt.gca().get_legend_handles_labels()
labels = ['Finn et al. (vision)', 'vision+haptic', 'vision+haptic+audio', 'vision+haptic+audio+vibro']
order = [3,2,1,0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper left')

plt.savefig('sep_behave_group_bar.png', dpi=300)
plt.show()
pass