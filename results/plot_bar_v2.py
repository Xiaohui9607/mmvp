import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

path = './sepvsall'

sns.set(style="darkgrid")
# sns.set(rc={'figure.figsize':(8.7,11.27)})
dpi = 300
# f, axes = plt.subplots(1, 2, figsize=(16, 5))
plt.rc('ytick', labelsize=16)
plt.rc('xtick', labelsize=16)
plt.rc('axes', labelsize=16)
# plt.rc('xtick', labelsize=16)
fig = plt.figure(figsize=(3400/300.0, 1800/300.0))
behaviors = ['crush', 'lift', 'grasp', 'shake', 'push', 'tap', 'hold', 'drop', 'poke']
settings = ['baseline', 'haptic', 'haptic_audio', 'haptic_audio_vibro']
xlabels = ['Finn et al.', 'vision+haptic', 'vision+haptic\n+audio', 'vision+haptic\n+audio+vibro']
plist = []
for i, setting in enumerate(settings):
    xs = []
    ys = []
    dfs = []
    for j, behave in enumerate(behaviors):
        df = pd.read_csv(os.path.join(path, setting, '{}.csv'.format(behave)))
        dfs.append(df.iloc[:, 1])
    # ax = axes[i]
    p = pd.DataFrame(dfs).T
    p.columns=behaviors
    p1 = p.melt(var_name='setting', value_name='SSIM')
    p1 = p1.dropna(axis=0,how='any')
    p2=p1.loc[:31778].copy()
    p2['setting'] = xlabels[settings.index(setting)]
    plist.append(p2)
    # p3=pd.concat([p1, P2],axis=0)
    # pass
    # sns.barplot(x='behavior', y='value', data=p1, ax=ax, ci="sd")
p = pd.concat(plist, axis=0)
sns.barplot(x='setting', y='SSIM', data=p, ci= 'sd')

# plt.rcParams.update({'font.size': 16})
plt.ylim([0.7, 0.9])
# ax.set_title('+' +'+'.join(setting.split('_')))
plt.savefig('sep_behave_horizontal_bar.png', dpi=300)
plt.show()
pass