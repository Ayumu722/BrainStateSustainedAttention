# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017 1016

@author: ayumu
"""

#%%
import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.cluster import AgglomerativeClustering
import scipy.io
Z = norm.ppf

def CalculateEffect(data1,data2):
    n1 = len(data1)
    n2 = len(data2)
    x1 = np.mean(data1)
    x2 = np.mean(data2)
    s1 = np.std(data1)
    s2 = np.std(data2)
    sd = np.sqrt((s1**2+s2**2)/2)
    s = np.std(data1-data2)
    
    g = abs(x1-x2)/sd
    biasFac = np.sqrt((n1-2)/(n1-1))
    g_unbiased = g*biasFac
    return g_unbiased

##############
# parameters #
##############
top_dir = 'C:/Users/ayumu/Dropbox/gradCPT/code/BrainStateSustainedAttention/'
data_dir = top_dir + 'data/Dataset1/'
basin_dir = data_dir + 'energylandscape/rest/'

# demographic data
demo = pd.read_csv(glob.glob(data_dir + '/participants.tsv')[0],delimiter='\t')
subs = demo['participants_id']

## roi
roi = 'Schaefer400_7Net'
net_order = ['DefaultMode', 'Limbic', 'PrefrontalControl','DorsalAttention','Salience','SomatoMotor','Visual']
roi_dir = top_dir + 'Parcellations/'
network = list(pd.read_csv(roi_dir + roi + '.txt',header=None)[0])

mat = scipy.io.loadmat(basin_dir + '/LocalMin_Summary.mat')
tmp = np.reshape(mat['vectorList'][:,mat['LocalMinIndex']-1],[len(mat['vectorList']),len(mat['LocalMinIndex'])])
brain_activity_pattern = pd.DataFrame(tmp,index=network)
brain_activity_pattern.columns = ['State 1','State 2']
brain_activity_pattern = brain_activity_pattern.reindex(index=net_order)

all_patterns = np.reshape(mat['vectorList'],[mat['vectorList'].shape[0],mat['vectorList'].shape[1]])
all_brain_activity_pattern = pd.DataFrame(all_patterns,index=network)
all_brain_activity_pattern = all_brain_activity_pattern.reindex(index=net_order)

metric='hamming'
method='complete'

fig0 = plt.figure(figsize=(12,6))
fig0 = sns.heatmap(all_brain_activity_pattern.iloc[:,mat['AdjacentList'][mat['LocalMinIndex'][0][0]-1]-1],cbar = False,cmap='Pastel1_r', linewidths=.3)
fig0.set_xticklabels([e for row in np.round(mat['E'],2)[mat['AdjacentList'][mat['LocalMinIndex'][0][0]-1]-1] for e in row], fontsize=15)
fig0.set_xlabel('Energy')

fig1 = plt.figure(figsize=(12,6))
fig1 = sns.heatmap(all_brain_activity_pattern.iloc[:,mat['AdjacentList'][mat['LocalMinIndex'][1][0]-1]-1],cbar = False,cmap='Pastel1_r', linewidths=.3)
fig1.set_xticklabels([e for row in np.round(mat['E'],2)[mat['AdjacentList'][mat['LocalMinIndex'][1][0]-1]-1] for e in row], fontsize=15)
fig1.set_xlabel('Energy')

MyPalette = ["#67a9cf","#ef8a62"]
n_clusters=2
ac = AgglomerativeClustering(n_clusters=n_clusters,
                            affinity=metric,
                            linkage=method)
cluster = ac.fit_predict(brain_activity_pattern.T)

# extract signals
DATA = pd.DataFrame()
for sub_i in subs:
    data_files = glob.glob(basin_dir + '/*' + sub_i + '*_BN.csv');data_files.sort()
    DATA_sub = pd.DataFrame()
    state_run = pd.DataFrame()
    for file_i in data_files:
        data = pd.read_csv(file_i,header=None)
        state_run = state_run.append(data)
    DATA_sub['state'] = state_run[0]
    DATA_sub['subid'] = sub_i        
    DATA = DATA.append(DATA_sub)

DATA['summary_state'] = np.zeros(DATA['state'].shape)
state_all = pd.unique(DATA.state)
state_all.sort()
cluster.sort()
for state_i in state_all:
    DATA.summary_state = np.where(DATA.state == state_i,cluster[state_i-1],DATA.summary_state)

num_state = len(pd.unique(DATA.state))
NEWDATA = DATA.groupby(['subid'])['state'].value_counts().unstack().fillna(0).apply(lambda x:100*x/sum(x),axis=1)
DATA_scat = pd.melt(NEWDATA)
df =pd.DataFrame({'MEAN':NEWDATA.mean(),'summary_state':cluster}).reset_index()
df['summary_state2'] = np.where(df.summary_state==0,'State 1','State 2')
df.index= ['State 1','State 2']

plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
sns.heatmap(brain_activity_pattern, cbar = False,cmap='Pastel1_r', linewidths=.3)

plt.subplot(1,2,2)
fig=sns.stripplot(x='state',y='value',data=DATA_scat,jitter=0,color='black',alpha=0.3)
fig=plt.plot([0,1],NEWDATA.T,color='black',alpha=0.3)
fig=sns.barplot(y='MEAN',x='summary_state2',hue = 'summary_state2',
                  dodge=False,data=df,order=['State 1','State 2'],hue_order=['State 1','State 2'],palette=MyPalette)
fig.set_ylim(0,65)
fig.set_ylabel("Percentage of total time", fontsize=15)
fig.legend('')
fig.set_xlabel("", fontsize=15) 

d = CalculateEffect(DATA_scat.query('state==1')["value"].reset_index(drop=True),DATA_scat.query('state==2')["value"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat.query('state==1')["value"],DATA_scat.query('state==2')["value"]),d)

