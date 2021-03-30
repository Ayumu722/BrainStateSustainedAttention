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
    
    return(out)

def MakeData(DATA):
    NEWVTCDATA = pd.DataFrame()
    NEWVTCDATA['State 1'] = DATA.query('summary_state==0').groupby(['subid'])['VarianceTimeCourse'].mean()
    NEWVTCDATA['State 2'] = DATA.query('summary_state==1').groupby(['subid'])['VarianceTimeCourse'].mean()
    DATA_scat_vtc = pd.melt(NEWVTCDATA)

    ## comparing of RT
    NEWDATA_RT = pd.DataFrame()
    NEWDATA_RT['State 1'] = DATA.query('summary_state==0').groupby(['subid'])['ReactionTime_Interpolated'].mean()
    NEWDATA_RT['State 2'] = DATA.query('summary_state==1').groupby(['subid'])['ReactionTime_Interpolated'].mean()
    DATA_scat_RT = pd.melt(NEWDATA_RT)

    ## comparing of MW
    NEWDATA_MW = pd.DataFrame()
    NEWDATA_MW['State 1'] = DATA.query('summary_state==0').groupby(['subid'])['ThoughtProbe_Interpolated'].mean()
    NEWDATA_MW['State 2'] = DATA.query('summary_state==1').groupby(['subid'])['ThoughtProbe_Interpolated'].mean()
    DATA_scat_MW = pd.melt(NEWDATA_MW)

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
    return DATA_scat_vtc, DATA_scat_RT, DATA_scat_dprime, DATA_scat_MW

##############
# parameters #
##############
tr=1.08
performance_list = ['CommissionError', 'CorrectCommission', 'CorrectOmission', 'OmissionError', 'In_the_Zone', 'VarianceTimeCourse','ReactionTime_Interpolated','ThoughtProbe_Interpolated']

top_dir = 'C:/Users/ayumu/Dropbox/gradCPT/code/BrainStateSustainedAttention/'
roi_dir = top_dir + 'Parcellations/'

data_hc_dir = top_dir + 'data/Dataset2/'
events_hc_dir = data_hc_dir + 'events/'
basin_hc_dir = data_hc_dir + 'energylandscape/'
subs_hc =  pd.read_csv(glob.glob(data_hc_dir + '/participants_HC.tsv')[0],delimiter='\t')['participants_id']
sub_num_hc = len(subs_hc)


data_adhd_dir = top_dir + 'data/Dataset3/'
events_adhd_dir = data_adhd_dir + 'events/'
basin_adhd_dir = data_adhd_dir + 'energylandscape/'
subs_adhd =  pd.read_csv(glob.glob(data_adhd_dir + '/participants_ADHD.tsv')[0],delimiter='\t')['participants_id']
sub_num_adhd = len(subs_adhd)

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

mat = scipy.io.loadmat(basin_hc_dir + roi + '/LocalMin_Summary.mat')
tmp = np.reshape(mat['vectorList'][:,mat['LocalMinIndex']-1],[len(mat['vectorList']),len(mat['LocalMinIndex'])])
brain_activity_pattern_HC = pd.DataFrame(tmp,index=network)
brain_activity_pattern_HC.columns = ['State 1','State 2']
brain_activity_pattern_HC = brain_activity_pattern_HC.reindex(index=net_order)
all_patterns = np.reshape(mat['vectorList'],[mat['vectorList'].shape[0],mat['vectorList'].shape[1]])
all_brain_activity_pattern_HC = pd.DataFrame(all_patterns,index=network)
all_brain_activity_pattern_HC = all_brain_activity_pattern_HC.reindex(index=net_order)

fig0 = plt.figure(figsize=(12,6))
fig0 = sns.heatmap(all_brain_activity_pattern_HC.iloc[:,mat['AdjacentList'][mat['LocalMinIndex'][0][0]-1]-1],cbar = False,cmap='Pastel1_r', linewidths=.3)
fig0.set_xticklabels([e for row in np.round(mat['E'],2)[mat['AdjacentList'][mat['LocalMinIndex'][0][0]-1]-1] for e in row], fontsize=15)
fig0.set_xlabel('Energy')

fig1 = plt.figure(figsize=(12,6))
fig1 = sns.heatmap(all_brain_activity_pattern_HC.iloc[:,mat['AdjacentList'][mat['LocalMinIndex'][1][0]-1]-1],cbar = False,cmap='Pastel1_r', linewidths=.3)
fig1.set_xticklabels([e for row in np.round(mat['E'],2)[mat['AdjacentList'][mat['LocalMinIndex'][1][0]-1]-1] for e in row], fontsize=15)
fig1.set_xlabel('Energy')

mat = scipy.io.loadmat(basin_adhd_dir + roi + '/LocalMin_Summary.mat')
tmp = np.reshape(mat['vectorList'][:,mat['LocalMinIndex']-1],[len(mat['vectorList']),len(mat['LocalMinIndex'])])
brain_activity_pattern_ADHD = pd.DataFrame(tmp,index=network)
brain_activity_pattern_ADHD = pd.DataFrame(tmp,index=network)
brain_activity_pattern_ADHD.columns = ['State 1','State 2']
brain_activity_pattern_ADHD = brain_activity_pattern_ADHD.reindex(index=net_order)
all_patterns = np.reshape(mat['vectorList'],[mat['vectorList'].shape[0],mat['vectorList'].shape[1]])
all_brain_activity_pattern_ADHD = pd.DataFrame(all_patterns,index=network)
all_brain_activity_pattern_ADHD = all_brain_activity_pattern_ADHD.reindex(index=net_order)

fig0 = plt.figure(figsize=(12,6))
fig0 = sns.heatmap(all_brain_activity_pattern_ADHD.iloc[:,mat['AdjacentList'][mat['LocalMinIndex'][0][0]-1]-1],cbar = False,cmap='Pastel1_r', linewidths=.3)
fig0.set_xticklabels([e for row in np.round(mat['E'],2)[mat['AdjacentList'][mat['LocalMinIndex'][0][0]-1]-1] for e in row], fontsize=15)
fig0.set_xlabel('Energy')

fig1 = plt.figure(figsize=(12,6))
fig1 = sns.heatmap(all_brain_activity_pattern_ADHD.iloc[:,mat['AdjacentList'][mat['LocalMinIndex'][1][0]-1]-1],cbar = False,cmap='Pastel1_r', linewidths=.3)
fig1.set_xticklabels([e for row in np.round(mat['E'],2)[mat['AdjacentList'][mat['LocalMinIndex'][1][0]-1]-1] for e in row], fontsize=15)
fig1.set_xlabel('Energy')

cluster_HC = ac.fit_predict(brain_activity_pattern_HC.T)
cluster_ADHD = ac.fit_predict(brain_activity_pattern_ADHD.T)


# extract signals
DATA_HC = pd.DataFrame()
for sub_i in subs_hc:
    task_files = glob.glob(events_hc_dir + sub_i +'*task-gradCPTMW*events.tsv');task_files.sort()
    data_files = glob.glob(basin_hc_dir + roi + '/*' + sub_i + '*_BN.csv');data_files.sort()

    DATA_sub = pd.DataFrame()
    state_run = pd.Series()
    sessions = pd.Series()
    for num_file_i,file_i in enumerate(task_files):
        taskinfo = pd.read_csv(file_i,delimiter='\t')
        data = pd.read_csv(data_files[num_file_i],header=None)
        num_vol = data.shape[0]
        for onset_time in taskinfo['onset']:
            belong = CheckWhere(num_vol,tr,onset_time)
            state_run = state_run.append(data[belong].iloc[0])
            sessions = sessions.append(pd.Series(num_file_i+1))    
            
        DATA_sub = DATA_sub.append(taskinfo.loc[:,performance_list])
    DATA_sub['state'] = state_run.values
    DATA_sub['session'] = sessions.values
    DATA_sub['subid'] = sub_i        
    DATA_HC = DATA_HC.append(DATA_sub)

DATA_HC['summary_state'] = np.zeros(DATA_HC['state'].shape)
state_all = pd.unique(DATA_HC.state)
state_all.sort()
cluster_HC.sort()
for state_i in state_all:
    DATA_HC.summary_state = np.where(DATA_HC.state == state_i,cluster_HC[int(state_i)-1],DATA_HC.summary_state)

## figure of Duration
num_state_HC = len(pd.unique(DATA_HC.state))
NEWDATA = DATA_HC.groupby(['subid'])['state'].value_counts().unstack().fillna(0).apply(lambda x:100*x/sum(x),axis=1)
DATA_scat = pd.melt(NEWDATA)
df_HC =pd.DataFrame({'MEAN':NEWDATA.mean(),'summary_state':cluster_HC}).reset_index()
df_HC['summary_state2'] = np.where(df_HC.summary_state==0,'State 1','State 2')

plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
sns.heatmap(brain_activity_pattern_HC, cbar = False,cmap='Pastel1_r', linewidths=.3)
plt.subplot(1,2,2)
fig=sns.stripplot(x='state',y='value',data=DATA_scat,jitter=0,color='black',alpha=0.3)
fig=plt.plot([0,1],NEWDATA.T,color='black',alpha=0.3)
fig=sns.barplot(y='MEAN',x='summary_state2',hue = 'summary_state2',
                  dodge=False,data=df_HC,order=['State 1','State 2'],hue_order=['State 1','State 2'],palette=MyPalette)
fig.set_ylim(0,65)
fig.set_ylabel("Percentage of total time", fontsize=15)
fig.legend('')
fig.set_xlabel("", fontsize=15) 

## State comparison
ZONEDATA_HC = DATA_HC.groupby(['subid','summary_state'])['In_the_Zone'].value_counts().unstack().fillna(0).apply(lambda x:sum(x),axis=1).unstack().fillna(0).apply(lambda x:100*x/sum(x),axis=1)
NEWDATA_HC = pd.DataFrame()
NEWDATA_HC['State 1'] = ZONEDATA_HC[0]
NEWDATA_HC['State 2'] = ZONEDATA_HC[1]
DATA_scat_hc = pd.melt(NEWDATA_HC)
DATA_scat_hc['DIAGNOSIS'] = np.matlib.repmat('HC', sub_num_hc*2,1)
DATA_scat_hc['SUBID'] = np.matlib.repmat(subs_hc, 1,2)[0]

# extract signals
DATA_ADHD = pd.DataFrame()
for sub_i in subs_adhd:
    task_files = glob.glob(events_adhd_dir + sub_i +'*task-gradCPTMW*events.tsv');task_files.sort()
    data_files = glob.glob(basin_adhd_dir + roi + '/*' + sub_i + '*_BN.csv');data_files.sort()
    DATA_sub = pd.DataFrame()
    state_run = pd.Series()
    sessions = pd.Series()
    for num_file_i,file_i in enumerate(task_files):
        taskinfo = pd.read_csv(file_i,delimiter='\t')
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
    DATA_ADHD = DATA_ADHD.append(DATA_sub)

DATA_ADHD['summary_state'] = np.zeros(DATA_ADHD['state'].shape)
state_all = pd.unique(DATA_ADHD.state)
state_all.sort()
cluster_ADHD.sort()
for state_i in state_all:
    DATA_ADHD.summary_state = np.where(DATA_ADHD.state == state_i,cluster_ADHD[int(state_i)-1],DATA_ADHD.summary_state)

## figure of Duration
num_state_ADHD = len(pd.unique(DATA_ADHD.state))
NEWDATA = DATA_ADHD.groupby(['subid'])['state'].value_counts().unstack().fillna(0).apply(lambda x:100*x/sum(x),axis=1)
DATA_scat = pd.melt(NEWDATA)
df_ADHD =pd.DataFrame({'MEAN':NEWDATA.mean(),'summary_state':cluster_ADHD}).reset_index()
df_ADHD['summary_state2'] = np.where(df_ADHD.summary_state==0,'State 1','State 2')
plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
sns.heatmap(brain_activity_pattern_ADHD, cbar = False,cmap='Pastel1_r', linewidths=.3)
plt.subplot(1,2,2)
fig=sns.stripplot(x='state',y='value',data=DATA_scat,jitter=0,color='black',alpha=0.3)
fig=plt.plot([0,1],NEWDATA.T,color='black',alpha=0.3)
fig=sns.barplot(y='MEAN',x='summary_state2',hue = 'summary_state2',
                  dodge=False,data=df_ADHD,order=['State 1','State 2'],hue_order=['State 1','State 2'],palette=MyPalette)
fig.set_ylim(0,65)
fig.set_ylabel("Percentage of total time", fontsize=15)
fig.legend('')
fig.set_xlabel("", fontsize=15) 

## State comparison
ZONEDATA_ADHD = DATA_ADHD.groupby(['subid','summary_state'])['In_the_Zone'].value_counts().unstack().fillna(0).apply(lambda x:sum(x),axis=1).unstack().fillna(0).apply(lambda x:100*x/sum(x),axis=1)
NEWDATA_ADHD = pd.DataFrame()
NEWDATA_ADHD['State 1'] = ZONEDATA_ADHD[0]
NEWDATA_ADHD['State 2'] = ZONEDATA_ADHD[1]
DATA_scat_adhd = pd.melt(NEWDATA_ADHD)
DATA_scat_adhd['DIAGNOSIS'] = np.matlib.repmat('ADHD', sub_num_adhd*2,1)
DATA_scat_adhd['SUBID'] = np.matlib.repmat(subs_adhd, 1,2)[0]


DATA_scat = pd.concat([DATA_scat_hc,DATA_scat_adhd])

DATA_scat_vtc_ADHD, DATA_scat_RT_ADHD, DATA_scat_dprime_ADHD,DATA_scat_MW_ADHD = MakeData(DATA_ADHD)
DATA_scat_vtc_HC, DATA_scat_RT_HC, DATA_scat_dprime_HC,DATA_scat_MW_HC = MakeData(DATA_HC)

DATA_scat['VTC'] = np.append(DATA_scat_vtc_HC.value,DATA_scat_vtc_ADHD.value,axis=0)
DATA_scat['RT'] = np.append(DATA_scat_RT_HC.value,DATA_scat_RT_ADHD.value,axis=0)
DATA_scat['dprime'] = np.append(DATA_scat_dprime_HC.value,DATA_scat_dprime_ADHD.value,axis=0)
DATA_scat['MW'] = np.append(DATA_scat_MW_HC.value,DATA_scat_MW_ADHD.value,axis=0)
DATA_scat['MW'] = 100-DATA_scat['MW']

f = plt.figure(figsize=(20,12))
plt.subplots_adjust(wspace=0.4)
plt.subplot(1,5,1)
ax = sns.barplot(y="value",x="DIAGNOSIS", hue = "variable",data=DATA_scat, palette=MyPalette,ci=None)
ax1 = sns.stripplot(x='DIAGNOSIS',y='value', hue = "variable",data=DATA_scat,jitter=0,color='black',alpha=0.3,split = True)
ax1.legend_ = None
ax.set_ylabel("Percentage of total time in each brain state", fontsize=20) #title
ax.set_xlabel("") #title
ax.set_ylim(0, 60)
plt.subplot(1,5,2)
ax = sns.barplot(y="VTC",x="DIAGNOSIS", hue = "variable",data=DATA_scat, palette=MyPalette,ci=None)
ax1 = sns.stripplot(x='DIAGNOSIS',y='VTC', hue = "variable",data=DATA_scat,jitter=0,color='black',alpha=0.3,split = True)
ax1.legend_ = None
ax.set_ylabel("Variance time course", fontsize=20) #title
ax.set_xlabel("") #title
ax.set_ylim(0.7, 0.9)
plt.subplot(1,5,3)
ax = sns.barplot(y="RT",x="DIAGNOSIS", hue = "variable",data=DATA_scat, palette=MyPalette,ci=None)
ax1 = sns.stripplot(x='DIAGNOSIS',y='RT', hue = "variable",data=DATA_scat,jitter=0,color='black',alpha=0.3,split = True)
ax1.legend_ = None
ax.set_ylabel("Reaction time", fontsize=20) #title
ax.set_xlabel("") #title
ax.set_ylim(0.9, 1.6)
plt.subplot(1,5,4)
ax = sns.barplot(y="dprime",x="DIAGNOSIS", hue = "variable",data=DATA_scat, palette=MyPalette,ci=None)
ax1 = sns.stripplot(x='DIAGNOSIS',y='dprime', hue = "variable",data=DATA_scat,jitter=0,color='black',alpha=0.3,split = True)
ax1.legend_ = None
ax.set_ylabel("d prime", fontsize=20) #title
ax.set_xlabel("") #title
plt.rc('xtick',labelsize=20)
ax.set_ylim(0, 6)
plt.subplot(1,5,5)
ax = sns.barplot(y="MW",x="DIAGNOSIS", hue = "variable",data=DATA_scat, palette=MyPalette,ci=None)
ax1 = sns.stripplot(x='DIAGNOSIS',y='MW', hue = "variable",data=DATA_scat,jitter=0,color='black',alpha=0.3,split = True)
ax1.legend_ = None
ax.set_ylabel("MW score", fontsize=20) #title
ax.set_xlabel("") #title
plt.rc('xtick',labelsize=20)

import statsmodels.api as sm
model = sm.MixedLM.from_formula("value ~ variable*DIAGNOSIS", DATA_scat, groups=DATA_scat["SUBID"]).fit(reml=False).summary()
# model = sm.MixedLM.from_formula("VTC ~ variable*DIAGNOSIS", DATA_scat, groups=DATA_scat["SUBID"]).fit(reml=False).summary()
# model = sm.MixedLM.from_formula("RT ~ variable*DIAGNOSIS", DATA_scat, groups=DATA_scat["SUBID"]).fit(reml=False).summary()
# model = sm.MixedLM.from_formula("dprime ~ variable*DIAGNOSIS", DATA_scat, groups=DATA_scat["SUBID"]).fit(reml=False).summary()
# model = sm.MixedLM.from_formula("MW ~ variable*DIAGNOSIS", DATA_scat, groups=DATA_scat["SUBID"]).fit(reml=False).summary()
print('p value for interaction was %s' %model.tables[1].iloc[3,3])
print(model)

d_state_hc = CalculateEffect(DATA_scat_hc.query('variable=="State 1"')["value"].reset_index(drop=True),DATA_scat_hc.query('variable=="State 2"')["value"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat_hc.query('variable=="State 1"')["value"],DATA_scat_hc.query('variable=="State 2"')["value"]),d_state_hc)

d_state_adhd = CalculateEffect(DATA_scat_adhd.query('variable=="State 1"')["value"].reset_index(drop=True),DATA_scat_adhd.query('variable=="State 2"')["value"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat_adhd.query('variable=="State 1"')["value"],DATA_scat_adhd.query('variable=="State 2"')["value"]),d_state_adhd)

d_VTC = CalculateEffect(DATA_scat_vtc_ADHD.query('variable=="State 1"')["value"].reset_index(drop=True),DATA_scat_vtc_ADHD.query('variable=="State 2"')["value"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat_vtc_ADHD.query('variable=="State 1"')["value"],DATA_scat_vtc_ADHD.query('variable=="State 2"')["value"]),d_VTC)

d_RT = CalculateEffect(DATA_scat_RT_ADHD.query('variable=="State 1"')["value"].reset_index(drop=True),DATA_scat_RT_ADHD.query('variable=="State 2"')["value"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat_RT_ADHD.query('variable=="State 1"')["value"],DATA_scat_RT_ADHD.query('variable=="State 2"')["value"]),d_RT)

d_dprime = CalculateEffect(DATA_scat_dprime_ADHD.query('variable=="State 1"')["value"].reset_index(drop=True),DATA_scat_dprime_ADHD.query('variable=="State 2"')["value"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat_dprime_ADHD.query('variable=="State 1"')["value"],DATA_scat_dprime_ADHD.query('variable=="State 2"')["value"]),d_dprime)

d_mw = CalculateEffect(DATA_scat_MW_ADHD.query('variable=="State 1"')["value"].reset_index(drop=True),DATA_scat_MW_ADHD.query('variable=="State 2"')["value"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat_MW_ADHD.query('variable=="State 1"')["value"],DATA_scat_MW_ADHD.query('variable=="State 2"')["value"]),d_mw)
