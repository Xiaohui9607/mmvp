import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

path = './sepvsall'

sns.set(style="darkgrid")
# sns.set(rc={'figure.figsize':(8.7,11.27)})

# f, axes = plt.subplots(1, 2, figsize=(16, 5))

plt.rcParams.update({'font.size': 20})
plt.rc('ytick', labelsize=14)
plt.rc('xtick', labelsize=14)
f, axes = plt.subplots(1, 3, figsize=(28.9,6.5))
tips = sns.load_dataset("tips")
behaviors = ['crush', 'lift', 'grasp', 'shake', 'push', 'tap', 'hold', 'drop', 'poke', 'all']
settings = ['haptic', 'haptic_audio', 'haptic_audio_vibro']

for i, setting in enumerate(settings):
    xs = []
    ys = []
    dfs = []
    for j, behave in enumerate(behaviors):
        df = pd.read_csv(os.path.join(path, 'use_{}'.format(setting), '{}.csv'.format(behave)))
        dfs.append(df.iloc[:, 2])
    ax = axes[i]
    p = pd.DataFrame(dfs).T
    p.columns=behaviors
    p1 = p.melt(var_name='behavior', value_name='SSIM')
    p1 = p1.dropna(axis=0,how='any')
    P2=p1.loc[:31778].copy()
    P2['behavior']='ensemble'
    p3=pd.concat([p1, P2],axis=0)
    pass
    # sns.barplot(x='behavior', y='value', data=p1, ax=ax, ci="sd")
    sns.barplot(x='behavior', y='SSIM', data=p3, ax=ax, ci="sd")

    ax.set_ylim([0.7, 1])
    ax.set_title('+' +'+'.join(setting.split('_')))
plt.savefig('all_behave.png', dpi=300, bbox_inches='tight')
plt.show()
pass