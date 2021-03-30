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

top_dir = 'C:/Users/ayumu/Dropbox/gradCPT/code/BrainStateSustainedAttention/'

## roi
roi = 'Schaefer400_7Net'
roi_dir = top_dir + 'Parcellations/'
network = list(pd.read_csv(roi_dir + roi + '.txt',header=None)[0])

org_dir = top_dir + 'data/Dataset1/energylandscape/' + roi
rest_dir = top_dir + 'data/Dataset1/energylandscape/rest/'
hc_dir = top_dir + 'data/Dataset2/energylandscape/' + roi
adhd_dir = top_dir + 'data/Dataset3/energylandscape/' + roi
reward_dir = top_dir + 'data/Dataset4/energylandscape/' + roi + '/reward/'
nonreward_dir = top_dir + 'data/Dataset4/energylandscape/' + roi + '/nonreward/'

org = scipy.io.loadmat(org_dir + '/LocalMin_Summary.mat')
all_patterns = np.reshape(org['vectorList'],[org['vectorList'].shape[0],org['vectorList'].shape[1]])
all_brain_activity_pattern = pd.DataFrame(all_patterns,index=network)
all_brain_activity_pattern = all_brain_activity_pattern.reindex(index=['DefaultMode', 'Limbic', 'PrefrontalControl','DorsalAttention','Salience','SomatoMotor','Visual'])

fig0 = plt.figure(figsize=(12,6))
fig0 = sns.heatmap(all_brain_activity_pattern.iloc[:,org['AdjacentList'][org['LocalMinIndex'][0][0]-1]-1],cbar = False,cmap='Pastel1_r', linewidths=.3)
fig0.set_xticklabels([e for row in np.round(org['E'],2)[org['AdjacentList'][org['LocalMinIndex'][0][0]-1]-1] for e in row], fontsize=15)
fig0.set_xlabel('Energy')

fig1 = plt.figure(figsize=(12,6))
fig1 = sns.heatmap(all_brain_activity_pattern.iloc[:,org['AdjacentList'][org['LocalMinIndex'][1][0]-1]-1],cbar = False,cmap='Pastel1_r', linewidths=.3)
fig1.set_xticklabels([e for row in np.round(org['E'],2)[org['AdjacentList'][org['LocalMinIndex'][1][0]-1]-1] for e in row], fontsize=15)
fig1.set_xlabel('Energy')

rest = scipy.io.loadmat(rest_dir  + '/LocalMin_Summary.mat')
hc = scipy.io.loadmat(hc_dir  + '/LocalMin_Summary.mat')
adhd = scipy.io.loadmat(adhd_dir  + '/LocalMin_Summary.mat')
reward = scipy.io.loadmat(reward_dir  + '/LocalMin_Summary.mat')
nonreward = scipy.io.loadmat(nonreward_dir  + '/LocalMin_Summary.mat')

df = pd.DataFrame()
df = df.append(pd.DataFrame({'energy':np.round(org['E'],2)[org['AdjacentList'][org['LocalMinIndex'][0][0]-1]-1].flatten(),'data':'Original'}))
df = df.append(pd.DataFrame({'energy':np.round(rest['E'],2)[rest['AdjacentList'][org['LocalMinIndex'][0][0]-1]-1].flatten(),'data':'Rest'}))
df = df.append(pd.DataFrame({'energy':np.round(hc['E'],2)[hc['AdjacentList'][org['LocalMinIndex'][0][0]-1]-1].flatten(),'data':'Replication'}))
df = df.append(pd.DataFrame({'energy':np.round(adhd['E'],2)[adhd['AdjacentList'][org['LocalMinIndex'][0][0]-1]-1].flatten(),'data':'ADHD'}))
df = df.append(pd.DataFrame({'energy':np.round(reward['E'],2)[reward['AdjacentList'][org['LocalMinIndex'][0][0]-1]-1].flatten(),'data':'Reward'}))
df = df.append(pd.DataFrame({'energy':np.round(nonreward['E'],2)[nonreward['AdjacentList'][org['LocalMinIndex'][0][0]-1]-1].flatten(),'data':'nonReward'}))
df = df.reset_index()

df_state2 = pd.DataFrame()
df_state2 = df_state2.append(pd.DataFrame({'energy':np.round(org['E'],2)[org['AdjacentList'][org['LocalMinIndex'][1][0]-1]-1].flatten(),'data':'Original'}))
df_state2 = df_state2.append(pd.DataFrame({'energy':np.round(rest['E'],2)[rest['AdjacentList'][org['LocalMinIndex'][1][0]-1]-1].flatten(),'data':'Rest'}))
df_state2 = df_state2.append(pd.DataFrame({'energy':np.round(hc['E'],2)[hc['AdjacentList'][org['LocalMinIndex'][1][0]-1]-1].flatten(),'data':'Replication'}))
df_state2 = df_state2.append(pd.DataFrame({'energy':np.round(adhd['E'],2)[adhd['AdjacentList'][org['LocalMinIndex'][1][0]-1]-1].flatten(),'data':'ADHD'}))
df_state2 = df_state2.append(pd.DataFrame({'energy':np.round(reward['E'],2)[reward['AdjacentList'][org['LocalMinIndex'][1][0]-1]-1].flatten(),'data':'Reward'}))
df_state2 = df_state2.append(pd.DataFrame({'energy':np.round(nonreward['E'],2)[nonreward['AdjacentList'][org['LocalMinIndex'][1][0]-1]-1].flatten(),'data':'nonReward'}))
df_state2 = df_state2.reset_index()


MyPalette = ["#67a9cf","#ef8a62"]
sns.set_context('paper')

plt.figure(figsize=([8,5]))
sns.barplot(data=df,x="index",y="energy",color=MyPalette[0],alpha=.5,ci=None)
sns.stripplot(x="index", y="energy",hue='data',
              data=df, dodge=.1)

plt.figure(figsize=([8,5]))
sns.barplot(data=df_state2,x="index",y="energy",color=MyPalette[1],alpha=.5,ci=None)
sns.stripplot(x="index", y="energy",hue='data',
              data=df_state2, dodge=.1)
