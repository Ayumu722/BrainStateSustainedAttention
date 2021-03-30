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
from numpy import matlib as mb

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

def MakeData(DATA,outname):
    NEWVTCDATA = pd.DataFrame()
    NEWVTCDATA['State 1'] = DATA.query('summary_state==0').groupby(['subid'])['VarianceTimeCourse'].mean()
    NEWVTCDATA['State 2'] = DATA.query('summary_state==1').groupby(['subid'])['VarianceTimeCourse'].mean()
    DATA_scat_vtc = pd.melt(NEWVTCDATA)

    ## comparing of RT
    NEWDATA_RT = pd.DataFrame()
    NEWDATA_RT['State 1'] = DATA.query('summary_state==0').groupby(['subid'])['ReactionTime_Interpolated'].mean()
    NEWDATA_RT['State 2'] = DATA.query('summary_state==1').groupby(['subid'])['ReactionTime_Interpolated'].mean()
    DATA_scat_RT = pd.melt(NEWDATA_RT)

    DATA_DPRIME = pd.DataFrame()
    DATA_DPRIME['HIT_DMN'] = DATA.query('summary_state==0').groupby(['subid'])['CorrectOmission'].value_counts().unstack()[1].fillna(0)
    DATA_DPRIME['MISS_DMN'] = DATA.query('summary_state==0').groupby(['subid'])['CommissionError'].value_counts().unstack()[1].fillna(0)
    DATA_DPRIME['CR_DMN'] = DATA.query('summary_state==0').groupby(['subid'])['CorrectCommission'].value_counts().unstack()[1].fillna(0)
    DATA_DPRIME['FA_DMN'] = DATA.query('summary_state==0').groupby(['subid'])['OmissionError'].value_counts().unstack()[1].fillna(0)
    DATA_DPRIME['HIT_DAN'] = DATA.query('summary_state==1').groupby(['subid'])['CorrectOmission'].value_counts().unstack()[1].fillna(0)
    DATA_DPRIME['MISS_DAN'] = DATA.query('summary_state==1').groupby(['subid'])['CommissionError'].value_counts().unstack()[1].fillna(0)
    DATA_DPRIME['CR_DAN'] = DATA.query('summary_state==1').groupby(['subid'])['CorrectCommission'].value_counts().unstack()[1].fillna(0)
    DATA_DPRIME['FA_DAN'] = DATA.query('summary_state==1').groupby(['subid'])['OmissionError'].value_counts().unstack()[1].fillna(0)
    
    dprime_DMN = []
    dprime_DAN = []
    for numsub_i, sub_i in enumerate(pd.unique(DATA.subid)):
        out_dmn = SDT(DATA_DPRIME['HIT_DMN'][numsub_i],DATA_DPRIME['MISS_DMN'][numsub_i],DATA_DPRIME['FA_DMN'][numsub_i],DATA_DPRIME['CR_DMN'][numsub_i])
        dprime_DMN.append(out_dmn['d'])
        out_dan = SDT(DATA_DPRIME['HIT_DAN'][numsub_i],DATA_DPRIME['MISS_DAN'][numsub_i],DATA_DPRIME['FA_DAN'][numsub_i],DATA_DPRIME['CR_DAN'][numsub_i])
        dprime_DAN.append(out_dan['d'])
    
    NEWDATA_DPRIME = pd.DataFrame()
    NEWDATA_DPRIME['State 1'] = dprime_DMN
    NEWDATA_DPRIME['State 2'] = dprime_DAN
    DATA_scat_dprime = pd.melt(NEWDATA_DPRIME)
    return DATA_scat_vtc, DATA_scat_RT, DATA_scat_dprime

##############
# parameters #
##############
tr=2.0
performance_list = ['CommissionError', 'CorrectCommission', 'CorrectOmission', 'OmissionError', 'In_the_Zone', 'VarianceTimeCourse','ReactionTime_Interpolated']

top_dir = 'C:/Users/ayumu/Dropbox/gradCPT/code/BrainStateSustainedAttention/'
data_dir = top_dir + 'data/Dataset4/'
events_dir = data_dir + 'events/'
basin_dir = data_dir + 'energylandscape/'
roi_dir = top_dir + 'Parcellations/'

# demographic data
demo = pd.read_csv(glob.glob(data_dir + 'participants_reward.tsv')[0],delimiter='\t')
subs = demo['participants_id']

net_order = ['DefaultMode', 'Limbic', 'PrefrontalControl','DorsalAttention','Salience','SomatoMotor','Visual']
roi = 'Schaefer400_7Net'
# roi = 'Schaefer400_8Net'
# roi = 'Yeo_7Net'
network = list(pd.read_csv(roi_dir + roi + '.txt',header=None)[0])

metric='hamming'
method='complete'

MyPalette = ["#67a9cf","#ef8a62"]
n_clusters=2
ac = AgglomerativeClustering(n_clusters=n_clusters,
                            affinity=metric,
                            linkage=method)

mat = scipy.io.loadmat(basin_dir + roi + '/reward/LocalMin_Summary.mat')
tmp = np.reshape(mat['vectorList'][:,mat['LocalMinIndex']-1],[len(mat['vectorList']),len(mat['LocalMinIndex'])])
brain_activity_pattern_reward = pd.DataFrame(tmp,index=network)
brain_activity_pattern_reward.columns = ['State 1','State 2']
brain_activity_pattern_reward = brain_activity_pattern_reward.reindex(index=net_order)
all_patterns = np.reshape(mat['vectorList'],[mat['vectorList'].shape[0],mat['vectorList'].shape[1]])
all_brain_activity_pattern_reward = pd.DataFrame(all_patterns,index=network)
all_brain_activity_pattern_reward = all_brain_activity_pattern_reward.reindex(index=net_order)

fig0 = plt.figure(figsize=(12,6))
fig0 = sns.heatmap(all_brain_activity_pattern_reward.iloc[:,mat['AdjacentList'][mat['LocalMinIndex'][0][0]-1]-1],cbar = False,cmap='Pastel1_r', linewidths=.3)
fig0.set_xticklabels([e for row in np.round(mat['E'],2)[mat['AdjacentList'][mat['LocalMinIndex'][0][0]-1]-1] for e in row], fontsize=15)
fig0.set_xlabel('Energy')

fig1 = plt.figure(figsize=(12,6))
fig1 = sns.heatmap(all_brain_activity_pattern_reward.iloc[:,mat['AdjacentList'][mat['LocalMinIndex'][1][0]-1]-1],cbar = False,cmap='Pastel1_r', linewidths=.3)
fig1.set_xticklabels([e for row in np.round(mat['E'],2)[mat['AdjacentList'][mat['LocalMinIndex'][1][0]-1]-1] for e in row], fontsize=15)
fig1.set_xlabel('Energy')

mat = scipy.io.loadmat(basin_dir + roi + '/nonreward/LocalMin_Summary.mat')
tmp = np.reshape(mat['vectorList'][:,mat['LocalMinIndex']-1],[len(mat['vectorList']),len(mat['LocalMinIndex'])])
brain_activity_pattern_nonreward = pd.DataFrame(tmp,index=network)
brain_activity_pattern_nonreward.columns = ['State 1','State 2']
brain_activity_pattern_nonreward = brain_activity_pattern_nonreward.reindex(index=net_order)
all_patterns = np.reshape(mat['vectorList'],[mat['vectorList'].shape[0],mat['vectorList'].shape[1]])
all_brain_activity_pattern_nonreward = pd.DataFrame(all_patterns,index=network)
all_brain_activity_pattern_nonreward = all_brain_activity_pattern_nonreward.reindex(index=net_order)

fig0 = plt.figure(figsize=(12,6))
fig0 = sns.heatmap(all_brain_activity_pattern_nonreward.iloc[:,mat['AdjacentList'][mat['LocalMinIndex'][0][0]-1]-1],cbar = False,cmap='Pastel1_r', linewidths=.3)
fig0.set_xticklabels([e for row in np.round(mat['E'],2)[mat['AdjacentList'][mat['LocalMinIndex'][0][0]-1]-1] for e in row], fontsize=15)
fig0.set_xlabel('Energy')

fig1 = plt.figure(figsize=(12,6))
fig1 = sns.heatmap(all_brain_activity_pattern_nonreward.iloc[:,mat['AdjacentList'][mat['LocalMinIndex'][1][0]-1]-1],cbar = False,cmap='Pastel1_r', linewidths=.3)
fig1.set_xticklabels([e for row in np.round(mat['E'],2)[mat['AdjacentList'][mat['LocalMinIndex'][1][0]-1]-1] for e in row], fontsize=15)
fig1.set_xlabel('Energy')

cluster_reward = ac.fit_predict(brain_activity_pattern_reward.T)
cluster_nonreward = ac.fit_predict(brain_activity_pattern_nonreward.T)

# demographic data
REWARD = pd.DataFrame({'reward1':demo.reward1,'reward2':demo.reward2,'reward3':demo.reward3,'reward4':demo.reward4,'reward5':demo.reward5})
REWARD.index=subs

# extract signals
DATA_reward = pd.DataFrame()
DATA_nonreward = pd.DataFrame()
start_vol_reward = 0
start_vol_nonreward = 6

for num_sub_i,sub_i in enumerate(subs):
    task_files = glob.glob(events_dir + sub_i +'*task-gradCPT*events.tsv');task_files.sort()
    data_reward_files = glob.glob(basin_dir + roi + '/reward/*' + sub_i + '*_BN.csv');data_reward_files.sort()
    data_nonreward_files = glob.glob(basin_dir + roi + '/nonreward/*' + sub_i + '*_BN.csv');data_nonreward_files.sort()
    DATA_sub_reward = pd.DataFrame()
    DATA_sub_nonreward = pd.DataFrame()
    state_run_reward = pd.Series()
    state_run_nonreward = pd.Series()
    for num_file_i,file_i in enumerate(task_files):
        taskinfo = pd.read_csv(file_i,delimiter='\t')
        data_reward = pd.read_csv(data_reward_files[num_file_i],header=None)
        data_nonreward = pd.read_csv(data_nonreward_files[num_file_i],header=None)
        RewardStart = int(REWARD.iloc[num_sub_i,num_file_i])
        RewardTrial = np.zeros(len(taskinfo),)
        num_vol = len(data_reward)+len(data_nonreward)
        RewardVolume = np.zeros(num_vol,)
        nonRewardVolume = np.zeros(num_vol,)
        StateVolume = np.zeros(num_vol,)
        onset = taskinfo['onset']
        if RewardStart:
            RewardTrial[0:75] = 1
            RewardTrial[150:225] = 1
            RewardTrial[300:375] = 1
            RewardTrial[450:525] = 1
        else:
            RewardTrial[75:150] = 1
            RewardTrial[225:300] = 1
            RewardTrial[375:450] = 1
            RewardTrial[525:len(taskinfo)+1] = 1
        for num_onset_i,onset_time in enumerate(onset):
            belong = CheckWhere(num_vol,tr,onset_time)
            RewardVolume[belong] = RewardTrial[num_onset_i]
            nonRewardVolume[belong] = abs(RewardTrial[num_onset_i]-1)
            if RewardVolume[belong]:
                if len(data_reward)<=start_vol_reward+int(sum(RewardVolume))-1:
                    state_run_reward = state_run_reward.append(pd.Series(data_reward[0][start_vol_reward+int(sum(RewardVolume))-2]))
                else:
                    state_run_reward = state_run_reward.append(pd.Series(data_reward[0][start_vol_reward+int(sum(RewardVolume))-1]))
                    
            else:
                if len(data_nonreward)<=start_vol_nonreward+int(sum(nonRewardVolume))-1:
                    state_run_nonreward = state_run_nonreward.append(pd.Series(data_nonreward[0][start_vol_nonreward+int(sum(nonRewardVolume))-2]))
                else:
                    state_run_nonreward = state_run_nonreward.append(pd.Series(data_nonreward[0][start_vol_nonreward+int(sum(nonRewardVolume))-1]))                   
 
        DATA_sub_reward = DATA_sub_reward.append(taskinfo.loc[RewardTrial==1,performance_list])
        DATA_sub_nonreward = DATA_sub_nonreward.append(taskinfo.loc[RewardTrial==0,performance_list])
    DATA_sub_reward['state'] = state_run_reward.values
    DATA_sub_reward['subid'] = sub_i        
    DATA_sub_nonreward['state'] = state_run_nonreward.values
    DATA_sub_nonreward['subid'] = sub_i  
    DATA_reward = DATA_reward.append(DATA_sub_reward)
    DATA_nonreward = DATA_nonreward.append(DATA_sub_nonreward)

DATA_reward['summary_state'] = np.zeros(DATA_reward['state'].shape)
state_all = pd.unique(DATA_reward.state)
state_all.sort()
cluster_reward.sort()
for state_i in state_all:
    DATA_reward.summary_state = np.where(DATA_reward.state == state_i,cluster_reward[int(state_i)-1],DATA_reward.summary_state)
    
DATA_nonreward['summary_state'] = np.zeros(DATA_nonreward['state'].shape)
state_all = pd.unique(DATA_nonreward.state)
state_all.sort()
cluster_nonreward.sort()
for state_i in state_all:
    DATA_nonreward.summary_state = np.where(DATA_nonreward.state == state_i,cluster_nonreward[int(state_i)-1],DATA_nonreward.summary_state)

## figure of Duration
num_state_reward = len(pd.unique(DATA_reward.state))
NEWDATA = DATA_reward.groupby(['subid'])['state'].value_counts().unstack().fillna(0).apply(lambda x:100*x/sum(x),axis=1)
DATA_scat = pd.melt(NEWDATA)
df_reward =pd.DataFrame({'MEAN':NEWDATA.mean(),'summary_state':cluster_reward}).reset_index()
df_reward['summary_state2'] = np.where(df_reward.summary_state==0,'State 1','State 2')

plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
sns.heatmap(brain_activity_pattern_reward, cbar = False,cmap='Pastel1_r', linewidths=.3)
plt.subplot(1,2,2)
fig=sns.stripplot(x='state',y='value',data=DATA_scat,jitter=0,color='black',alpha=0.3)
fig=plt.plot([0,1],NEWDATA.T,color='black',alpha=0.3)
fig=sns.barplot(y='MEAN',x='summary_state2',hue = 'summary_state2',
                  dodge=False,data=df_reward,order=['State 1','State 2'],hue_order=['State 1','State 2'],palette=MyPalette)
fig.set_ylim(0,60)
fig.set_ylabel("Percentage of total time", fontsize=15)
fig.legend('')
fig.set_xlabel("", fontsize=15) 

## figure of Duration
num_state_nonreward = len(pd.unique(DATA_nonreward.state))
NEWDATA = DATA_nonreward.groupby(['subid'])['state'].value_counts().unstack().fillna(0).apply(lambda x:100*x/sum(x),axis=1)
DATA_scat = pd.melt(NEWDATA)
df_nonreward =pd.DataFrame({'MEAN':NEWDATA.mean(),'summary_state':cluster_nonreward}).reset_index()
df_nonreward['summary_state2'] = np.where(df_nonreward.summary_state==0,'State 1','State 2')
plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
sns.heatmap(brain_activity_pattern_nonreward, cbar = False,cmap='Pastel1_r', linewidths=.3)
plt.subplot(1,2,2)
fig=sns.stripplot(x='state',y='value',data=DATA_scat,jitter=0,color='black',alpha=0.3)
fig=plt.plot([0,1],NEWDATA.T,color='black',alpha=0.3)
fig=sns.barplot(y='MEAN',x='summary_state2',hue = 'summary_state2',
                  dodge=False,data=df_nonreward,order=['State 1','State 2'],hue_order=['State 1','State 2'],palette=MyPalette)
fig.set_ylim(0,60)
fig.set_ylabel("Percentage of total time", fontsize=15)
fig.legend('')
fig.set_xlabel("", fontsize=15) 

DATA_scat_vtc_reward, DATA_scat_RT_reward, DATA_scat_dprime_reward = MakeData(DATA_reward,'Reward')
DATA_scat_vtc_nonreward, DATA_scat_RT_nonreward, DATA_scat_dprime_nonreward = MakeData(DATA_nonreward,'nonReward')

## State comparison
ZONEDATA_reward = DATA_reward.groupby(['subid','summary_state'])['In_the_Zone'].value_counts().unstack().fillna(0).apply(lambda x:sum(x),axis=1).unstack().fillna(0).apply(lambda x:100*x/sum(x),axis=1)
ZONEDATA_nonreward = DATA_nonreward.groupby(['subid','summary_state'])['In_the_Zone'].value_counts().unstack().fillna(0).apply(lambda x:sum(x),axis=1).unstack().fillna(0).apply(lambda x:100*x/sum(x),axis=1)
NEWDATA = pd.DataFrame()
NEWDATA['State1_reward'] = ZONEDATA_reward[0]
NEWDATA['State2_reward'] = ZONEDATA_reward[1]
NEWDATA['State1_nonreward'] =ZONEDATA_nonreward[0]
NEWDATA['State2_nonreward'] = ZONEDATA_nonreward[1]
DATA_scat = pd.melt(NEWDATA)

STATE = np.matlib.repmat(('State 1','State 2'), 16,2).T.reshape(16*4,)
REWARD = np.matlib.repmat(('Reward','nonReward'), 32,1).T.reshape(32*2,)
SUBID = np.matlib.repmat(subs,1,4)
DATA_scat['SUBID'] = SUBID[0]
DATA_scat['STATE'] = STATE
DATA_scat['REWARD'] = REWARD
DATA_scat['VTC'] = np.append(DATA_scat_vtc_reward.value,DATA_scat_vtc_nonreward.value,axis=0)
DATA_scat['RT'] = np.append(DATA_scat_RT_reward.value,DATA_scat_RT_nonreward.value,axis=0)
DATA_scat['dprime'] = np.append(DATA_scat_dprime_reward.value,DATA_scat_dprime_nonreward.value,axis=0)

print(stats.mannwhitneyu(DATA_scat.query('variable=="State1_reward"')["value"],DATA_scat.query('variable=="State1_nonreward"')["value"]))
print(stats.mannwhitneyu(DATA_scat.query('variable=="State2_reward"')["value"],DATA_scat.query('variable=="State2_nonreward"')["value"]))

f = plt.figure(figsize=(20,12))
plt.subplots_adjust(wspace=0.4)
plt.subplot(1,4,1)
ax = sns.barplot(y="value",x="REWARD", hue = "STATE",data=DATA_scat, palette=MyPalette,ci=None)
ax1 = sns.stripplot(x='REWARD',y='value', hue = "STATE",data=DATA_scat,jitter=0,color='black',alpha=0.3,split = True)
ax1.legend_ = None
ax.set_ylabel("Percentage of total time in each brain state", fontsize=20) #title
ax.set_xlabel("") #title
ax.set_ylim(0, 60)
plt.subplot(1,4,2)
ax = sns.barplot(y="VTC",x="REWARD", hue = "STATE",data=DATA_scat, palette=MyPalette,ci=None)
ax1 = sns.stripplot(x='REWARD',y='VTC', hue = "STATE",data=DATA_scat,jitter=0,color='black',alpha=0.3,split = True)
ax1.legend_ = None
ax.set_ylabel("Variance time course", fontsize=20) #title
ax.set_xlabel("") #title
ax.set_ylim(0.5, 0.9)
plt.subplot(1,4,3)
ax = sns.barplot(y="RT",x="REWARD", hue = "STATE",data=DATA_scat, palette=MyPalette,ci=None)
ax1 = sns.stripplot(x='REWARD',y='RT', hue = "STATE",data=DATA_scat,jitter=0,color='black',alpha=0.3,split = True)
ax1.legend_ = None
ax.set_ylabel("Reaction time", fontsize=20) #title
ax.set_xlabel("") #title
ax.set_ylim(0.55, 0.8)
plt.subplot(1,4,4)
ax = sns.barplot(y="dprime",x="REWARD", hue = "STATE",data=DATA_scat, palette=MyPalette,ci=None)
ax1 = sns.stripplot(x='REWARD',y='dprime', hue = "STATE",data=DATA_scat,jitter=0,color='black',alpha=0.3,split = True)
ax1.legend_ = None
ax.set_ylabel("d prime", fontsize=20) #title
ax.set_xlabel("") #title
plt.tick_params(labelsize=20)
ax.set_ylim(0, 6)

### Statistical analysis
import statsmodels.api as sm
model = sm.MixedLM.from_formula("value ~ STATE*REWARD", DATA_scat, groups=DATA_scat["SUBID"]).fit().summary()
# model = sm.MixedLM.from_formula("VTC ~ STATE*REWARD", DATA_scat, groups=DATA_scat["SUBID"]).fit(reml=False).summary()
# model = sm.MixedLM.from_formula("RT ~ STATE*REWARD", DATA_scat, groups=DATA_scat["SUBID"]).fit(reml=False).summary()
# model = sm.MixedLM.from_formula("dprime ~ STATE*REWARD", DATA_scat, groups=DATA_scat["SUBID"]).fit(reml=False).summary()
print('p value for interaction was %s' %model.tables[1].iloc[3,3])
print(model)

d_state_nonreward = CalculateEffect(DATA_scat.query('variable=="State1_nonreward"')["value"].reset_index(drop=True),DATA_scat.query('variable=="State2_nonreward"')["value"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat.query('variable=="State1_nonreward"')["value"],DATA_scat.query('variable=="State2_nonreward"')["value"]),d_state_nonreward)
d_state_reward = CalculateEffect(DATA_scat.query('variable=="State1_reward"')["value"].reset_index(drop=True),DATA_scat.query('variable=="State2_reward"')["value"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat.query('variable=="State1_reward"')["value"],DATA_scat.query('variable=="State2_reward"')["value"]),d_state_reward)

d_state = CalculateEffect(DATA_scat.query('variable=="State1_nonreward"')["value"].reset_index(drop=True),DATA_scat.query('variable=="State1_reward"')["value"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat.query('variable=="State1_nonreward"')["value"],DATA_scat.query('variable=="State1_reward"')["value"]),d_state)


d_VTC_nonreward = CalculateEffect(DATA_scat.query('variable=="State1_nonreward"')["VTC"].reset_index(drop=True),DATA_scat.query('variable=="State2_nonreward"')["VTC"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat.query('variable=="State1_nonreward"')["VTC"],DATA_scat.query('variable=="State2_nonreward"')["VTC"]),d_VTC_nonreward)
d_VTC_reward = CalculateEffect(DATA_scat.query('variable=="State1_reward"')["VTC"].reset_index(drop=True),DATA_scat.query('variable=="State2_reward"')["VTC"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat.query('variable=="State1_reward"')["VTC"],DATA_scat.query('variable=="State2_reward"')["VTC"]),d_VTC_reward)

d_RT_nonreward = CalculateEffect(DATA_scat.query('variable=="State1_nonreward"')["RT"].reset_index(drop=True),DATA_scat.query('variable=="State2_nonreward"')["RT"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat.query('variable=="State1_nonreward"')["RT"],DATA_scat.query('variable=="State2_nonreward"')["RT"]),d_RT_nonreward)
d_RT_reward = CalculateEffect(DATA_scat.query('variable=="State1_reward"')["RT"].reset_index(drop=True),DATA_scat.query('variable=="State2_reward"')["RT"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat.query('variable=="State1_reward"')["RT"],DATA_scat.query('variable=="State2_reward"')["RT"]),d_RT_reward)

d_dprime_nonreward = CalculateEffect(DATA_scat.query('variable=="State1_nonreward"')["dprime"].reset_index(drop=True),DATA_scat.query('variable=="State2_nonreward"')["dprime"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat.query('variable=="State1_nonreward"')["dprime"],DATA_scat.query('variable=="State2_nonreward"')["dprime"]),d_dprime_nonreward)
d_dprime_reward = CalculateEffect(DATA_scat.query('variable=="State1_reward"')["dprime"].reset_index(drop=True),DATA_scat.query('variable=="State2_reward"')["dprime"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat.query('variable=="State1_reward"')["dprime"],DATA_scat.query('variable=="State2_reward"')["dprime"]),d_dprime_reward)
