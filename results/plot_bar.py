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

tips = sns.load_dataset("tips")
behaviors = ['crush', 'lift', 'grasp', 'shake', 'push', 'tap', 'hold', 'drop', 'poke', 'all']
settings = ['haptic', 'haptic_audio', 'haptic_audio_vibro']

for i, setting in enumerate(settings):
    # plt.rcParams.update({'font.size': 20})
    plt.rc('ytick', labelsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('axes', labelsize=16)
    # f, axes = plt.subplots(1, 3, figsize=(28.9,6.5))
    fig = plt.figure(figsize=(3400/300.0, 1800/300.0))
    xs = []
    ys = []
    dfs = []
    for j, behave in enumerate(behaviors):
        df = pd.read_csv(os.path.join(path, setting, '{}.csv'.format(behave)))
        dfs.append(df.iloc[:, 1])
    # ax = axes[i]
    p = pd.DataFrame(dfs).T
    p.columns = behaviors
    p1 = p.melt(var_name='behavior', value_name='SSIM')
    p1 = p1.dropna(axis=0,how='any')
    P2 = p1.loc[:31778].copy()
    P2['behavior'] = 'ensemble'
    p3=pd.concat([p1, P2],axis=0)
    sns.barplot(x='behavior', y='SSIM', data=p3, ci="sd")
    # sns.boxplot(x='behavior', y='SSIM', data=p3)

    plt.ylim([0.6, 1])
    # ax.set_title('+' +'+'.join(setting.split('_')))
    plt.savefig('sep_behave_vertical_{}.png'.format(setting), dpi=300)
    plt.show()
    pass