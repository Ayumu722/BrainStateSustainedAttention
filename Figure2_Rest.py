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
import os
from scipy import stats
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
import math
from sklearn.cluster import AgglomerativeClustering
import scipy.io
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
Z = norm.ppf

def CheckWhere(num_vol,tr,onset_time):
    onset_time = onset_time + 5 # explain for hemodynamic response
    if onset_time>num_vol*tr: onset_time = num_vol*tr-1 # explain for hemodynamic response
    x = np.arange(tr, (num_vol+1)*tr,tr)
    belong = np.logical_and(onset_time <= x,onset_time>x-2)
    return belong

def SDT(hits, misses, fas, crs): # hits, misses, false alarms, correct rejection
    """ returns a dict with d-prime measures given hits, misses, false alarms, and correct rejections"""
    # Floors an ceilings are replaced by half hits and half FA's
    half_hit = 0.5 / (hits + misses)
    half_fa = 0.5 / (fas + crs)
 
    # Calculate hit_rate and avoid d' infinity
    hit_rate = hits / (hits + misses)
    if hit_rate == 1: 
        hit_rate = 1 - half_hit
    if hit_rate == 0: 
        hit_rate = half_hit
 
    # Calculate false alarm rate and avoid d' infinity
    fa_rate = fas / (fas + crs)
    if fa_rate == 1: 
        fa_rate = 1 - half_fa
    if fa_rate == 0: 
        fa_rate = half_fa
 
    # Return d', beta, c and Ad'
    out = {}
    out['d'] = Z(hit_rate) - Z(fa_rate)
    out['beta'] = math.exp((Z(fa_rate)**2 - Z(hit_rate)**2) / 2)
    out['c'] = -(Z(hit_rate) + Z(fa_rate)) / 2
    out['Ad'] = norm.cdf(out['d'] / math.sqrt(2))
    
    return(out)

##############
# parameters #
##############
dummy = 0
tr=2.0
save_flag = 0

performance_list = ['CommissionError', 'CorrectCommission', 'CorrectOmission', 'OmissionError', 'In_the_Zone', 'VarianceTimeCourse','ReactionTime_Interpolated']

top_dir = 'C:/Users/ayumu/Dropbox/gradCPT/'
basin_dir = top_dir + 'data/Rest/EnergyLandscape/'

# ## Schaefer400_7Net
# roi = 'Schaefer400_7Net'
# roi_dir = 'C:/Users/ayumu/Dropbox/gradCPT/Parcellations/' + roi + '/'
# ROI_files = pd.read_csv(roi_dir + 'Schaefer400_7Net.csv')
# roiname = ROI_files.Network
# network = np.unique(roiname)
# net_order = ['DefaultMode', 'Limbic', 'PrefrontalControl','DorsalAttention','Salience','SomatoMotor','Visual']

## Schaefer400_8Net
roi = 'Schaefer400_8Net'
roi_dir = 'C:/Users/ayumu/Dropbox/gradCPT/Parcellations/' + roi + '/'
ROI_files = pd.read_csv(roi_dir + 'Schaefer400_8Net.csv')
roiname = ROI_files.Network
network = np.unique(roiname)
net_order = ['DefaultMode', 'Limbic', 'PrefrontalControlB', 'PrefrontalControlA','DorsalAttention','Salience','SomatoMotor','Visual']

# ## Shen
# roi = 'Shen'
# roi_dir = top_dir + '/Parcellations/' + roi + '/'
# ROI_files = roi_dir + 'shen_2mm_268_parcellation.nii.gz'
# NET_files = pd.read_csv(roi_dir + 'Shen.txt',header=None)
# roiname = NET_files[0]
# network = roiname

# ## Power
# roi = 'Power'
# roi_dir = top_dir + '/Parcellations/' + roi + '/'
# ROI_files = pd.read_csv(roi_dir + 'Power.csv')
# roiname = ROI_files.Network
# network = np.unique(roiname)

# ## Power_7Net
# roi = 'Power_7Net'
# roi_dir = top_dir + '/Parcellations/' + roi + '/'
# ROI_files = pd.read_csv(roi_dir + roi + '.csv')
# roiname = ROI_files.Network
# network = np.unique(roiname)

# ## Power_11Net
# roi = 'Power_11Net'
# roi_dir = top_dir + '/Parcellations/' + roi + '/'
# ROI_files = pd.read_csv(roi_dir + roi + '.csv')
# roiname = ROI_files.Network
# network = np.unique(roiname)

# ## Yeo_7Net
# roi = 'Yeo_7Net'
# roi_dir = top_dir + '/Parcellations/' + roi + '/'
# ROI_files = roi_dir + 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz'
# NET_files = pd.read_csv(roi_dir + 'Yeo_7Net.csv')
# roiname = NET_files.Network
# network = np.unique(roiname)

# ## Yeo_Liberal_7Net
# roi = 'Yeo_Liberal_7Net'
# roi_dir = top_dir + '/Parcellations/' + roi + '/'
# ROI_files = roi_dir + 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz'
# NET_files = pd.read_csv(roi_dir + 'Yeo_Liberal_7Net.csv')
# roiname = NET_files.Network
# network = np.unique(roiname)

fig_dir = top_dir + 'fig/Rest/EnergyLandscape/' + roi
if os.path.isdir(fig_dir)==False: os.mkdir(fig_dir)

mat = scipy.io.loadmat(basin_dir + roi + '/LocalMin_Summary.mat')
tmp = np.reshape(mat['vectorList'][:,mat['LocalMinIndex']-1],[len(mat['vectorList']),len(mat['LocalMinIndex'])])
brain_activity_pattern = pd.DataFrame(tmp,index=network)
brain_activity_pattern.columns = ['State 1','State 2']
brain_activity_pattern = brain_activity_pattern.reindex(index=net_order)

all_patterns = np.reshape(mat['vectorList'],[mat['vectorList'].shape[0],mat['vectorList'].shape[1]])
all_brain_activity_pattern = pd.DataFrame(all_patterns,index=network)
all_brain_activity_pattern = all_brain_activity_pattern.reindex(index=net_order)

# metric='euclidean'
# metric='mahalanobis'
metric='hamming'

# method='single'
method='complete'
# method='average'
# method='ward'
# method='weighted'
# method='centroid'

y = pdist(brain_activity_pattern.T,metric='hamming')
# sns.clustermap(brain_activity_pattern,cbar = False,cmap='Pastel1', linewidths=.3)
sns.clustermap(brain_activity_pattern, row_cluster=False,
               method=method, metric=metric,
               cbar = False,cmap='Pastel1_r', linewidths=.3)
row_clusters = linkage(pdist(brain_activity_pattern.T,metric=metric),method=method)
row_dendr = dendrogram(row_clusters,no_plot=True)
if save_flag==1:plt.savefig(fig_dir + '/activation_pattern.pdf')
if save_flag==1:plt.savefig(fig_dir + '/activation_pattern.png')


fig0 = plt.figure(figsize=(12,6))
fig0 = sns.heatmap(all_brain_activity_pattern.iloc[:,mat['AdjacentList'][mat['LocalMinIndex'][0][0]-1]-1],cbar = False,cmap='Pastel1_r', linewidths=.3)
fig0.set_xticklabels([e for row in np.round(mat['E'],2)[mat['AdjacentList'][mat['LocalMinIndex'][0][0]-1]-1] for e in row], fontsize=15)
fig0.set_xlabel('Energy')
if save_flag==1:plt.savefig(fig_dir + '/Adjuscent_Energy_state1.pdf')
if save_flag==1:plt.savefig(fig_dir + '/Adjuscent_Energy_state1.png')

fig1 = plt.figure(figsize=(12,6))
fig1 = sns.heatmap(all_brain_activity_pattern.iloc[:,mat['AdjacentList'][mat['LocalMinIndex'][1][0]-1]-1],cbar = False,cmap='Pastel1_r', linewidths=.3)
fig1.set_xticklabels([e for row in np.round(mat['E'],2)[mat['AdjacentList'][mat['LocalMinIndex'][1][0]-1]-1] for e in row], fontsize=15)
fig1.set_xlabel('Energy')
if save_flag==1:plt.savefig(fig_dir + '/Adjuscent_Energy_state2.pdf')
if save_flag==1:plt.savefig(fig_dir + '/Adjuscent_Energy_state2.png')


MyPalette = ["#67a9cf","#ef8a62"]
n_clusters=2
ac = AgglomerativeClustering(n_clusters=n_clusters,
                            affinity=metric,
                            linkage=method)
cluster = ac.fit_predict(brain_activity_pattern.T)

# demographic data
demo = pd.read_csv(glob.glob(top_dir + '/code/participants.tsv')[0],delimiter='\t')
subs = demo['participants_id']

# extract signals
DATA = pd.DataFrame()
for sub_i in subs:
    data_files = glob.glob(basin_dir + roi + '/*' + sub_i + '*_BN.csv');data_files.sort()
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
NEWDATA.to_csv(top_dir + 'code/EnergyLandscape/ForPaper/DATA_rest.csv',columns=None)

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
if save_flag==1:plt.savefig(fig_dir + '/total_time_state_all.pdf')
if save_flag==1:plt.savefig(fig_dir + '/total_time_state_all.png')