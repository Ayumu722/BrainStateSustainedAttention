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
import scipy.io
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering


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
tr=2.0

performance_list = ['CommissionError', 'CorrectCommission', 'CorrectOmission', 'OmissionError', 'In_the_Zone', 'VarianceTimeCourse','ReactionTime_Interpolated']

top_dir = 'C:/Users/ayumu/Dropbox/gradCPT/code/BrainStateSustainedAttention/'
data_dir = top_dir + 'data/Dataset1/'
events_dir = data_dir + 'events/'
basin_dir = data_dir + 'energylandscape/'

subs = pd.read_csv(glob.glob(data_dir + '/participants.tsv')[0],delimiter='\t')['participants_id']

## roi
roi = 'Power'
roi_dir = top_dir + 'Parcellations/'
network = list(pd.read_csv(roi_dir + roi + '.txt',header=None)[0])

mat = scipy.io.loadmat(basin_dir + roi + '/LocalMin_Summary.mat')
tmp = np.reshape(mat['vectorList'][:,mat['LocalMinIndex']-1],[len(mat['vectorList']),len(mat['LocalMinIndex'])])
brain_activity_pattern = pd.DataFrame(tmp,index=network)
sns.heatmap(brain_activity_pattern,cbar = False,cmap='Pastel1_r', linewidths=.3)

# extract signals
DATA = pd.DataFrame()
for sub_i in subs:
    task_files = glob.glob(events_dir + sub_i +'*task-gradCPT*events.tsv');task_files.sort()
    data_files = glob.glob(basin_dir + roi + '/*' + sub_i + '*_BN.csv');data_files.sort()

    DATA_sub = pd.DataFrame()
    state_run = pd.DataFrame()
    sessions = pd.Series()
    for num_file_i,file_i in enumerate(task_files):
        taskinfo = pd.read_csv(file_i,delimiter='\t')
        data = pd.read_csv(data_files[num_file_i],header=None)
        num_vol = data.shape[0]
        for onset_time in taskinfo['onset']:
            belong = CheckWhere(num_vol,tr,onset_time)
            state_run = state_run.append(data[belong])
            sessions = sessions.append(pd.Series(num_file_i+1))    
        DATA_sub = DATA_sub.append(taskinfo.loc[:,performance_list])
    DATA_sub['state'] = state_run.values
    DATA_sub['subid'] = sub_i        
    DATA_sub['session'] = sessions.values        
    DATA = DATA.append(DATA_sub)

MyPalette = ["#67a9cf","#ef8a62"]
DATA['summary_state'] = np.zeros(DATA['state'].shape)
use_state = [6,1]
for num_i,state_i in enumerate(use_state,1):
    print(num_i)
    DATA.summary_state = np.where(DATA.state == state_i,'State%s' %num_i,DATA.summary_state)
state_all = ['State1','State2']

## comparing of VTC
NEWVTCDATA = pd.DataFrame()
for state_i in state_all:
    NEWVTCDATA[state_i] = DATA.query('summary_state==@state_i').groupby(['subid'])['VarianceTimeCourse'].mean()
DATA_scat_VTC = pd.melt(NEWVTCDATA)
f = plt.figure(figsize=(12,6))
plt.subplots_adjust(wspace=0.4)
plt.subplot(1,3,1)
ax = sns.barplot(y='value',x='variable', data=DATA_scat_VTC,ci=None,
                  dodge=False,order=['State2','State1'],palette=MyPalette)
sns.stripplot(x='variable',y='value',data=DATA_scat_VTC,order=['State2','State1'],jitter=0,color='black',alpha=0.3)
plt.plot([1,0],NEWVTCDATA.T,color='black',alpha=0.3)
ax.set_ylim(0.7,0.85)
ax.set_ylabel("Variance", fontsize=15) #title
ax.set_xlabel("") #title

## comparing of RT
NEWDATA_RT = pd.DataFrame()
for state_i in state_all:
    NEWDATA_RT[state_i] = DATA.query('summary_state==@state_i').groupby(['subid'])['ReactionTime_Interpolated'].mean()
DATA_scat_RT = pd.melt(NEWDATA_RT)
plt.subplot(1,3,2)
ax = sns.barplot(y='value',x='variable', data=DATA_scat_RT,ci=None,
                                   dodge=False,order=['State2','State1'],palette=MyPalette)
sns.stripplot(x='variable',y='value',data=DATA_scat_RT,order=['State2','State1'],jitter=0,color='black',alpha=0.3)
plt.plot([1,0],NEWDATA_RT.T,color='black',alpha=0.3)
ax.set_ylim(0.55,0.90)
ax.set_ylabel("Reaction time (s)", fontsize=15) #title
ax.set_xlabel("") #title

DATA_DPRIME = pd.DataFrame()
DATA_DPRIME['HIT_state1'] = DATA.query('summary_state=="State1"').groupby(['subid'])['CorrectOmission'].value_counts().unstack()[1].fillna(0)
DATA_DPRIME['MISS_state1'] = DATA.query('summary_state=="State1"').groupby(['subid'])['CommissionError'].value_counts().unstack()[1].fillna(0)
DATA_DPRIME['CR_state1'] = DATA.query('summary_state=="State1"').groupby(['subid'])['CorrectCommission'].value_counts().unstack()[1].fillna(0)
DATA_DPRIME['FA_state1'] = DATA.query('summary_state=="State1"').groupby(['subid'])['OmissionError'].value_counts().unstack()[1].fillna(0)

DATA_DPRIME['HIT_state2'] = DATA.query('summary_state=="State2"').groupby(['subid'])['CorrectOmission'].value_counts().unstack()[1].fillna(0)
DATA_DPRIME['MISS_state2'] = DATA.query('summary_state=="State2"').groupby(['subid'])['CommissionError'].value_counts().unstack()[1].fillna(0)
DATA_DPRIME['CR_state2'] = DATA.query('summary_state=="State2"').groupby(['subid'])['CorrectCommission'].value_counts().unstack()[1].fillna(0)
DATA_DPRIME['FA_state2'] = DATA.query('summary_state=="State2"').groupby(['subid'])['OmissionError'].value_counts().unstack()[1].fillna(0)


dprime1 = []
dprime2 = []
for numsub_i, sub_i in enumerate(subs):
    out_dmn = SDT(DATA_DPRIME['HIT_state1'][numsub_i],DATA_DPRIME['MISS_state1'][numsub_i],DATA_DPRIME['FA_state1'][numsub_i],DATA_DPRIME['CR_state1'][numsub_i])
    dprime1.append(out_dmn['d'])
    out_dan = SDT(DATA_DPRIME['HIT_state2'][numsub_i],DATA_DPRIME['MISS_state2'][numsub_i],DATA_DPRIME['FA_state2'][numsub_i],DATA_DPRIME['CR_state2'][numsub_i])
    dprime2.append(out_dan['d'])

NEWDATA_DPRIME = pd.DataFrame()
NEWDATA_DPRIME['State1'] = dprime1
NEWDATA_DPRIME['State2'] = dprime2
NEWDATA_DPRIME.index = subs
DATA_scat_dprime = pd.melt(NEWDATA_DPRIME)
plt.subplot(1,3,3)
ax = sns.barplot(y='value',x='variable', data=DATA_scat_dprime,ci=None,
                                   dodge=False,order=['State2','State1'],palette=MyPalette)
sns.stripplot(x='variable',y='value',data=DATA_scat_dprime,order=['State2','State1'],jitter=0,color='black',alpha=0.3)
plt.plot([1,0],NEWDATA_DPRIME.T,color='black',alpha=0.3)
ax.set_ylim(1.5,5)
ax.set_ylabel("d prime", fontsize=15) #title
ax.set_xlabel("") #title

print(stats.ttest_rel(DATA_scat_VTC.query('variable=="State1"')["value"],DATA_scat_VTC.query('variable=="State2"')["value"]))
print(stats.ttest_rel(DATA_scat_RT.query('variable=="State1"')["value"],DATA_scat_RT.query('variable=="State2"')["value"]))
print(stats.ttest_rel(DATA_scat_dprime.query('variable=="State1"')["value"],DATA_scat_dprime.query('variable=="State2"')["value"]))
