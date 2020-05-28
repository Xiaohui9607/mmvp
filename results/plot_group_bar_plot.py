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

fig = plt.figure(figsize=(3400/300.0, 1800/300.0))
behaviors = ['crush', 'lift', 'grasp', 'shake', 'push', 'tap', 'hold', 'drop', 'poke']
settings = ['baseline', 'haptic', 'haptic_audio', 'haptic_audio_vibro']
#xlabels = ['Finn et al.', 'vision+haptic', 'vision+haptic\n+audio', 'vision+haptic\n+audio+vibro']
plist = []

# add ensemble
dfs_baseline = []
dfs_haptic= []
dfs_haptic_audio = []
dfs_haptic_audio_vibro = []
# add ensemble

for i, behavior in enumerate(behaviors):
    xs = []
    ys = []
    dfs = []
    for j, a_setting in enumerate(settings):
        df = pd.read_csv(os.path.join(path, a_setting, '{}.csv'.format(behavior)))
        dfs.append(df.iloc[:, 2])
        
        # add ensemble
        if a_setting =='baseline':
            dfs_baseline.extend(df.iloc[:,2])
        elif a_setting == 'haptic':
            dfs_haptic.extend(df.iloc[:, 2])
        elif a_setting == 'haptic_audio':
            dfs_haptic_audio.extend(df.iloc[:, 2])
        else:
            dfs_haptic_audio_vibro.extend(df.iloc[:, 2])
        # add ensemble

    p = pd.DataFrame(dfs).T
    p.columns=settings
    p1 = p.melt(var_name='setting', value_name='SSIM')
    p1 = p1.dropna(axis=0,how='any')
    p1['behavior']= behavior
    plist.append(p1)

p = pd.concat(plist, axis=0)

# add ensemble
p_baseline = pd.DataFrame(dfs_baseline)
p_baseline.columns=['baseline']
p_baseline = p_baseline.melt(var_name='setting', value_name='SSIM')
p_baseline['behavior']= 'ensemble'

p_haptic = pd.DataFrame(dfs_haptic)
p_haptic.columns=['haptic']
p_haptic = p_haptic.melt(var_name='setting', value_name='SSIM')
p_haptic['behavior']= 'ensemble'

p_haptic_audio = pd.DataFrame(dfs_haptic_audio)
p_haptic_audio.columns=['haptic_audio']
p_haptic_audio = p_haptic_audio.melt(var_name='setting', value_name='SSIM')
p_haptic_audio['behavior']= 'ensemble'

p_haptic_audio_vibro = pd.DataFrame(dfs_haptic_audio_vibro)
p_haptic_audio_vibro.columns=['haptic_audio_vibro']
p_haptic_audio_vibro = p_haptic_audio_vibro.melt(var_name='setting', value_name='SSIM')
p_haptic_audio_vibro['behavior']= 'ensemble'
# add ensemble

p = pd.concat([p, p_baseline, p_haptic, p_haptic_audio, p_haptic_audio_vibro])

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