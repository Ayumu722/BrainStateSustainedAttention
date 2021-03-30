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
import math
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
    g = abs(x1-x2)/sd
    biasFac = np.sqrt((n1-2)/(n1-1))
    g_unbiased = g*biasFac
    return g_unbiased


def CheckWhere(num_vol,tr,onset_time):
    onset_time = onset_time + 5 # explain for hemodynamic response
    while onset_time>num_vol*tr: onset_time = num_vol*tr-1 # explain for hemodynamic response
    x = np.arange(tr, round((num_vol+1)*tr,2),tr)
    belong = np.logical_and(onset_time <= x,onset_time>x-1.5)
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
    out['hit_rate'] = hit_rate
    out['fa_rate'] = fa_rate
    
    return(out)


##############
# parameters #
##############
tr=1.08
performance_list = ['CommissionError', 'CorrectCommission', 'CorrectOmission', 'OmissionError', 'In_the_Zone', 'VarianceTimeCourse','ReactionTime_Interpolated','ThoughtProbe_Interpolated']
top_dir = 'C:/Users/ayumu/Dropbox/gradCPT/code/BrainStateSustainedAttention/'
data_dir = top_dir + 'data/Dataset2/'
events_dir = data_dir + 'events/'
basin_dir = data_dir + 'energylandscape/'
roi_dir = top_dir + 'Parcellations/'

# demographic data
demo = pd.read_csv(glob.glob(data_dir + 'participants_HC.tsv')[0],delimiter='\t')
subs = demo['participants_id']

roi = 'Schaefer400_7Net'
net_order = ['DefaultMode', 'Limbic', 'PrefrontalControl','DorsalAttention','Salience','SomatoMotor','Visual']

# roi = 'Schaefer400_8Net'
# net_order = ['DefaultMode', 'Limbic', 'PrefrontalControlB', 'PrefrontalControlA','DorsalAttention','Salience','SomatoMotor','Visual']

# roi = 'Yeo_7Net'
# net_order = ['DefaultMode', 'Limbic', 'PrefrontalControl','DorsalAttention','Salience','SomatoMotor','Visual']


network = list(pd.read_csv(roi_dir + roi + '.txt',header=None)[0])


mat = scipy.io.loadmat(basin_dir + roi + '/LocalMin_Summary.mat')
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
MW_sub = pd.DataFrame()
for sub_i in subs:
    task_files = glob.glob(events_dir + sub_i +'*task-gradCPTMW*events.tsv');task_files.sort()
    data_files = glob.glob(basin_dir + roi + '/*' + sub_i + '*_BN.csv');data_files.sort()

    DATA_sub = pd.DataFrame()
    state_run = pd.Series()
    sessions = pd.Series()
    mw_sessions = pd.Series()
    for num_file_i,file_i in enumerate(task_files):
        taskinfo = pd.read_csv(file_i,delimiter='\t')
        MW = taskinfo['ThoughtProbe']
        MW = MW.loc[np.where(MW)]
        mw_sessions = mw_sessions.append(pd.Series(MW))
        data = pd.read_csv(data_files[num_file_i],header=None)
        num_vol = data.shape[0]
        for onset_time in taskinfo['onset']:
            belong = CheckWhere(num_vol,tr,onset_time)
            state_run = state_run.append(data[belong].iloc[0])
            sessions = sessions.append(pd.Series(num_file_i+1))    
            
        DATA_sub = DATA_sub.append(taskinfo.loc[:,performance_list])
    DATA_sub['state'] = state_run.values
    DATA_sub['subid'] = sub_i        
    DATA_sub['session'] = sessions.values
    MW_sub = MW_sub.append(pd.DataFrame({'MW':mw_sessions.reset_index(drop=True),'subid':sub_i}))
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

## comparing of VTC
NEWVTCDATA = pd.DataFrame()
NEWVTCDATA['State 1'] = DATA.query('summary_state==0').groupby(['subid'])['VarianceTimeCourse'].mean()
NEWVTCDATA['State 2'] = DATA.query('summary_state==1').groupby(['subid'])['VarianceTimeCourse'].mean()
DATA_scat_VTC = pd.melt(NEWVTCDATA)
f = plt.figure(figsize=(12,6))
plt.subplots_adjust(wspace=0.4)
plt.subplot(1,4,1)
ax = sns.barplot(y='value',x='variable', data=DATA_scat_VTC,palette=MyPalette,ci=None)
sns.stripplot(x='variable',y='value',data=DATA_scat_VTC,jitter=0,color='black',alpha=0.3)
plt.plot([0,1],NEWVTCDATA.T,color='black',alpha=0.3)
ax.set_ylim(0.7,0.85)
ax.set_ylabel("Variance", fontsize=15) #title
ax.set_xlabel("") #title

## comparing of RT
NEWDATA_RT = pd.DataFrame()
NEWDATA_RT['State 1'] = DATA.query('summary_state==0').groupby(['subid'])['ReactionTime_Interpolated'].mean()
NEWDATA_RT['State 2'] = DATA.query('summary_state==1').groupby(['subid'])['ReactionTime_Interpolated'].mean()
DATA_scat_RT = pd.melt(NEWDATA_RT)
plt.subplot(1,4,2)
ax = sns.barplot(y='value',x='variable', data=DATA_scat_RT,palette=MyPalette,ci=None)
sns.stripplot(x='variable',y='value',data=DATA_scat_RT,jitter=0,color='black',alpha=0.3)
plt.plot([0,1],NEWDATA_RT.T,color='black',alpha=0.3)
ax.set_ylim(0.9,1.6)
ax.set_ylabel("Reaction time (s)", fontsize=15) #title
ax.set_xlabel("") #title

DATA_DPRIME = pd.DataFrame()
DATA_DPRIME['HIT_state1'] = DATA.query('summary_state==0').groupby(['subid'])['CorrectOmission'].value_counts().unstack()[1].fillna(0)
DATA_DPRIME['MISS_state1'] = DATA.query('summary_state==0').groupby(['subid'])['CommissionError'].value_counts().unstack()[1].fillna(0)
DATA_DPRIME['CR_state1'] = DATA.query('summary_state==0').groupby(['subid'])['CorrectCommission'].value_counts().unstack()[1].fillna(0)
DATA_DPRIME['FA_state1'] = DATA.query('summary_state==0').groupby(['subid'])['OmissionError'].value_counts().unstack()[1].fillna(0)

DATA_DPRIME['HIT_state2'] = DATA.query('summary_state==1').groupby(['subid'])['CorrectOmission'].value_counts().unstack()[1].fillna(0)
DATA_DPRIME['MISS_state2'] = DATA.query('summary_state==1').groupby(['subid'])['CommissionError'].value_counts().unstack()[1].fillna(0)
DATA_DPRIME['CR_state2'] = DATA.query('summary_state==1').groupby(['subid'])['CorrectCommission'].value_counts().unstack()[1].fillna(0)
DATA_DPRIME['FA_state2'] = DATA.query('summary_state==1').groupby(['subid'])['OmissionError'].value_counts().unstack()[1].fillna(0)


dprime1 = [];hit_rate1 = [];fa_rate1 = [];criterion1 = []
dprime2 = [];hit_rate2 = [];fa_rate2 = [];criterion2 = []

for numsub_i, sub_i in enumerate(subs):
    out_dmn = SDT(DATA_DPRIME['HIT_state1'][numsub_i],DATA_DPRIME['MISS_state1'][numsub_i],DATA_DPRIME['FA_state1'][numsub_i],DATA_DPRIME['CR_state1'][numsub_i])
    dprime1.append(out_dmn['d']);hit_rate1.append(out_dmn['hit_rate']);fa_rate1.append(out_dmn['fa_rate']);criterion1.append(out_dmn['c']);

    out_dan = SDT(DATA_DPRIME['HIT_state2'][numsub_i],DATA_DPRIME['MISS_state2'][numsub_i],DATA_DPRIME['FA_state2'][numsub_i],DATA_DPRIME['CR_state2'][numsub_i])
    dprime2.append(out_dan['d']);hit_rate2.append(out_dan['hit_rate']);fa_rate2.append(out_dan['fa_rate']);criterion2.append(out_dan['c']);


NEWDATA_DPRIME = pd.DataFrame()
NEWDATA_DPRIME['State 1'] = dprime1
NEWDATA_DPRIME['State 2'] = dprime2
NEWDATA_DPRIME.index = subs
DATA_scat_dprime = pd.melt(NEWDATA_DPRIME)
plt.subplot(1,4,3)
ax = sns.barplot(y=np.mean(NEWDATA_DPRIME),x=np.arange(2), data=DATA_scat_dprime, palette=MyPalette)
sns.stripplot(x='variable',y='value',data=DATA_scat_dprime,jitter=0,color='black',alpha=0.3)
plt.plot([0,1],NEWDATA_DPRIME.T,color='black',alpha=0.3)
ax.set_ylim(1.5,6)
ax.set_ylabel("d prime", fontsize=15) #title
ax.set_xlabel("") #title

## comparing of MW
NEWDATA_MW = pd.DataFrame()
NEWDATA_MW['State 1'] = DATA.query('summary_state==0').groupby(['subid'])['ThoughtProbe_Interpolated'].mean()
NEWDATA_MW['State 2'] = DATA.query('summary_state==1').groupby(['subid'])['ThoughtProbe_Interpolated'].mean()
NEWDATA_MW = 100-NEWDATA_MW
DATA_scat_MW = pd.melt(NEWDATA_MW)
plt.subplot(1,4,4)
ax = sns.barplot(y='value',x='variable', data=DATA_scat_MW,palette=MyPalette,ci=None)
sns.stripplot(x='variable',y='value',data=DATA_scat_MW,jitter=0,color='black',alpha=0.3)
plt.plot([0,1],NEWDATA_MW.T,color='black',alpha=0.3)
ax.set_ylim(0, 80)
ax.set_ylabel("Mind wandering score", fontsize=15) #title
ax.set_xlabel("") #title

d_VTC = CalculateEffect(DATA_scat_VTC.query('variable=="State 1"')["value"].reset_index(drop=True),DATA_scat_VTC.query('variable=="State 2"')["value"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat_VTC.query('variable=="State 1"')["value"],DATA_scat_VTC.query('variable=="State 2"')["value"]),d_VTC)

d_RT = CalculateEffect(DATA_scat_RT.query('variable=="State 1"')["value"].reset_index(drop=True),DATA_scat_RT.query('variable=="State 2"')["value"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat_RT.query('variable=="State 1"')["value"],DATA_scat_RT.query('variable=="State 2"')["value"]),d_RT)

d_dprime = CalculateEffect(DATA_scat_dprime.query('variable=="State 1"')["value"].reset_index(drop=True),DATA_scat_dprime.query('variable=="State 2"')["value"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat_dprime.query('variable=="State 1"')["value"],DATA_scat_dprime.query('variable=="State 2"')["value"]),d_dprime)

d_mw = CalculateEffect(DATA_scat_MW.query('variable=="State 1"')["value"].reset_index(drop=True),DATA_scat_MW.query('variable=="State 2"')["value"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat_MW.query('variable=="State 1"')["value"],DATA_scat_MW.query('variable=="State 2"')["value"]),d_mw)
