# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:18:55 2020

@author: ayumu
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.io
import os
import glob
import numpy.matlib as mbm
from scipy import stats

##############
# parameters #
##############

task = 'Dataset1'
# task = 'Dataset1_Rest'
# task = "Dataset2"
# task = "Dataset3"
# task = "Dataset4_Reward"
# task = "Dataset4_nonReward"

top_dir = 'C:/Users/ayumu/Dropbox/gradCPT/code/BrainStateSustainedAttention/'

## roi
roi = 'Schaefer400_7Net'
roi_dir = top_dir + 'Parcellations/'
network = list(pd.read_csv(roi_dir + roi + '.txt',header=None)[0])

# demographic data
if task == 'Dataset1':
    data_dir = top_dir + 'data/Dataset1/'
    basin_dir = data_dir + 'energylandscape/'
    subs = pd.read_csv(glob.glob(data_dir + '/participants.tsv')[0],delimiter='\t')['participants_id']
    mat = scipy.io.loadmat(basin_dir + roi + '/LocalMin_Summary.mat')
elif task == 'Dataset1_Rest':
    data_dir = top_dir + 'data/Dataset1/'
    subs = pd.read_csv(glob.glob(data_dir + '/participants.tsv')[0],delimiter='\t')['participants_id']
    basin_dir = data_dir + 'energylandscape/'
    mat = scipy.io.loadmat(basin_dir + roi + '/LocalMin_Summary.mat')
elif task == 'Dataset2':
    data_dir = top_dir + 'data/Dataset2/'
    subs = pd.read_csv(glob.glob(data_dir + '/participants_HC.tsv')[0],delimiter='\t')['participants_id']
    basin_dir = data_dir + 'energylandscape/'
    mat = scipy.io.loadmat(basin_dir + roi + '/LocalMin_Summary.mat')
elif task == 'Dataset3':
    data_dir = top_dir + 'data/Dataset3/'
    subs = pd.read_csv(glob.glob(data_dir + '/participants_ADHD.tsv')[0],delimiter='\t')['participants_id']
    basin_dir = data_dir + 'energylandscape/'
    mat = scipy.io.loadmat(basin_dir + roi + '/LocalMin_Summary.mat')
elif task == 'Dataset4_Reward':
    data_dir = top_dir + 'data/Dataset4/'
    subs = pd.read_csv(glob.glob(data_dir + '/participants_reward.tsv')[0],delimiter='\t')['participants_id']
    basin_dir = data_dir + 'energylandscape/'
    mat = scipy.io.loadmat(basin_dir + roi + '/reward/LocalMin_Summary.mat')
elif task == 'Dataset4_nonReward':
    data_dir = top_dir + 'data/Dataset4/'
    subs = pd.read_csv(glob.glob(data_dir + '/participants_reward.tsv')[0],delimiter='\t')['participants_id']
    basin_dir = data_dir + 'energylandscape/'
    mat = scipy.io.loadmat(basin_dir + roi + '/nonreward/LocalMin_Summary.mat')

sub_num = len(subs)

all_patterns = np.reshape(mat['vectorList'],[mat['vectorList'].shape[0],mat['vectorList'].shape[1]])
all_brain_activity_pattern = pd.DataFrame(all_patterns,index=network)
all_brain_activity_pattern = all_brain_activity_pattern.reindex(index=['DefaultMode', 'Limbic', 'PrefrontalControl','DorsalAttention','Salience','SomatoMotor','Visual'])


# extract signals
DATA = pd.DataFrame()
for sub_i in subs:
    if task == 'Dataset4_Reward':
        data_files = glob.glob(basin_dir + roi + '/reward/*' + sub_i + '*_SN.csv');data_files.sort()
    elif task == 'Dataset4_nonReward':
        data_files = glob.glob(basin_dir + roi + '/nonreward/*' + sub_i + '*_SN.csv');data_files.sort()
    else:
        data_files = glob.glob(basin_dir + roi + '/*' + sub_i + '*_SN.csv');data_files.sort()

    for num_file_i,file_i in enumerate(data_files):
        data = pd.read_csv(file_i,header=None)
        tmp = pd.DataFrame([sum(data[0]==i) for i in range(1,129)]).T
        tmp['session'] = num_file_i+1 
        tmp['subid'] = sub_i
        DATA = DATA.append(tmp)

DATA = DATA.groupby("subid").sum().reset_index()
DATA.index = DATA.subid
df_energy = pd.melt(DATA.T[1:129])
df_energy["pattern"] = mbm.repmat(list(range(128)),1,sub_num)[0]

df_energy_mean = pd.DataFrame(DATA.T[1:129].T.mean()).sort_values(0,ascending=False).reset_index()
df_energy_mean.columns = ["pattern","value"]
member_state1 = list([np.where(mat['BasinGraph'][:,2]==mat['LocalMinIndex'][0][0])][0][0])
member_state2 = list([np.where(mat['BasinGraph'][:,2]==mat['LocalMinIndex'][1][0])][0][0])

use_state1 = list(df_energy_mean.query("pattern==@member_state1")[0:10].pattern)
use_state2 = list(df_energy_mean.query("pattern==@member_state2")[0:10].pattern)

df_energy_state1 = df_energy.query("pattern==@use_state1").reset_index(drop=True)
df_energy_state2 = df_energy.query("pattern==@use_state2").reset_index(drop=True)


plt.figure(figsize=([5,5]))
plt.subplot(2,1,1)
sns.barplot(data=df_energy_state1,x="pattern",y="value",color='gray',order = use_state1,ci=None)
sns.stripplot(x="pattern", y="value",hue='subid',order = use_state1,
              data=df_energy_state1,color='black',alpha=.3, jitter=.1)
plt.legend('')
plt.subplot(2,1,2)
sns.heatmap(all_brain_activity_pattern.iloc[:,use_state1],cbar = False,cmap='Pastel1_r', linewidths=.3)
print(stats.ttest_rel(df_energy_state1.query("pattern==%s" %use_state1[0]).reset_index().value,df_energy_state1.query("pattern==%s" %use_state1[1]).reset_index().value))
print(stats.ttest_rel(df_energy_state1.query("pattern==%s" %use_state1[0]).reset_index().value,df_energy_state1.query("pattern==%s" %use_state1[2]).reset_index().value))
print(stats.ttest_rel(df_energy_state1.query("pattern==%s" %use_state1[1]).reset_index().value,df_energy_state1.query("pattern==%s" %use_state1[2]).reset_index().value))

plt.figure(figsize=([5,5]))
plt.subplot(2,1,1)
sns.barplot(data=df_energy_state2,x="pattern",y="value",color='gray',order = use_state2,ci=None)
sns.stripplot(x="pattern", y="value",hue='subid',order = use_state2,
              data=df_energy_state2,color='black',alpha=.3, jitter=.1)
plt.legend('')
plt.subplot(2,1,2)
sns.heatmap(all_brain_activity_pattern.iloc[:,use_state2],cbar = False,cmap='Pastel1_r', linewidths=.3)
print(stats.ttest_rel(df_energy_state2.query("pattern==%s" %use_state2[0]).reset_index().value,df_energy_state2.query("pattern==%s" %use_state2[1]).reset_index().value))
print(stats.ttest_rel(df_energy_state2.query("pattern==%s" %use_state2[0]).reset_index().value,df_energy_state2.query("pattern==%s" %use_state2[2]).reset_index().value))
print(stats.ttest_rel(df_energy_state2.query("pattern==%s" %use_state2[1]).reset_index().value,df_energy_state2.query("pattern==%s" %use_state2[2]).reset_index().value))

