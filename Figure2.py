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
import pingouin as pg

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
source_dir = top_dir + 'data/OriginalGradCPT/MRI/'
basin_dir = top_dir + 'data/OriginalGradCPT/EnergyLandscape/'

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
# NET_files = pd.read_csv(roi_dir + 'Yeo_7Net.txt',header=None)
# roiname = NET_files[0]
# network = list(roiname)

# ## Yeo_Liberal_7Net
# roi = 'Yeo_Liberal_7Net'
# roi_dir = top_dir + '/Parcellations/' + roi + '/'
# ROI_files = roi_dir + 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz'
# NET_files = pd.read_csv(roi_dir + 'Yeo_Liberal_7Net.txt',header=None)
# roiname = NET_files[0]
# network = list(roiname)

fig_dir = top_dir + 'fig/OriginalGradCPT/EnergyLandscape/' + roi
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

# fig0 = plt.figure(figsize=(8,16),facecolor='white')
# axd = plt.subplot(211)
# df_rowclust = brain_activity_pattern[row_dendr['leaves']].T
# axm = plt.subplot(212)
# sns.heatmap(df_rowclust.T,cmap="Pastel1",cbar = False)
# plt.pcolor(df_rowclust.T,cmap="Pastel1")
# plt.grid(False)

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
    nv_files = glob.glob(source_dir + sub_i + '*task-gradCPT_*_desc-confounds_regressors.tsv');nv_files.sort()
    task_files = glob.glob(source_dir + sub_i +'*task-gradCPT*events.tsv');task_files.sort()
    data_files = glob.glob(basin_dir + roi + '/*' + sub_i + '*_BN.csv');data_files.sort()

    DATA_sub = pd.DataFrame()
    state_run = pd.DataFrame()
    sessions = pd.Series()
    for num_file_i,file_i in enumerate(task_files):
        taskinfo = pd.read_csv(file_i,delimiter='\t')
        nv = pd.read_csv(nv_files[num_file_i], delimiter='\t')
        data = pd.read_csv(data_files[num_file_i],header=None)
        num_vol = nv.shape[0]
        for onset_time in taskinfo['onset']:
            belong = CheckWhere(num_vol,tr,onset_time)
            state_run = state_run.append(data[belong])
            sessions = sessions.append(pd.Series(num_file_i+1))    
        DATA_sub = DATA_sub.append(taskinfo.loc[:,performance_list])
    DATA_sub['state'] = state_run.values
    DATA_sub['subid'] = sub_i        
    DATA_sub['session'] = sessions.values        
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
NEWDATAVTC = DATA.groupby(['subid','state'])['VarianceTimeCourse'].mean().unstack().fillna(0)
DATA_scat_VTC = pd.melt(NEWDATAVTC)
df =pd.DataFrame({'MEAN':NEWDATA.mean(),'VTC':NEWDATAVTC.mean(),'summary_state':cluster}).reset_index()
df.state = df.state-1
df['summary_state2'] = np.where(df.summary_state==0,'State 1','State 2')

plt.figure(figsize=(6,4))
fig = sns.barplot(y='MEAN',x=np.arange(1,num_state+1),hue = 'summary_state2',
                  dodge=False,data=df.iloc[row_dendr['leaves']],
                  hue_order=['State 1','State 2'],palette=MyPalette)
fig.set_xticklabels(row_dendr['leaves'])
fig.set_ylim(0,60)
fig.set_ylabel("Percentage of total time", fontsize=15)
fig.legend(loc='upper right')
fig.set_xlabel("Individual state", fontsize=15) 
if save_flag==1:plt.savefig(fig_dir + '/total_time_state_all.pdf')
if save_flag==1:plt.savefig(fig_dir + '/total_time_state_all.png')

## comparing of VTC
NEWVTCDATA = pd.DataFrame()
NEWVTCDATA['State 1'] = DATA.query('summary_state==0').groupby(['subid'])['VarianceTimeCourse'].mean()
NEWVTCDATA['State 2'] = DATA.query('summary_state==1').groupby(['subid'])['VarianceTimeCourse'].mean()
DATA_scat_VTC = pd.melt(NEWVTCDATA)
f = plt.figure(figsize=(12,6))
plt.subplots_adjust(wspace=0.3)
plt.subplot(1,3,1)
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
plt.subplot(1,3,2)
ax = sns.barplot(y='value',x='variable', data=DATA_scat_RT,palette=MyPalette,ci=None)
sns.stripplot(x='variable',y='value',data=DATA_scat_RT,jitter=0,color='black',alpha=0.3)
plt.plot([0,1],NEWDATA_RT.T,color='black',alpha=0.3)
ax.set_ylim(0.55,0.9)
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


dprime1 = []
dprime2 = []
for numsub_i, sub_i in enumerate(subs):
    out_dmn = SDT(DATA_DPRIME['HIT_state1'][numsub_i],DATA_DPRIME['MISS_state1'][numsub_i],DATA_DPRIME['FA_state1'][numsub_i],DATA_DPRIME['CR_state1'][numsub_i])
    dprime1.append(out_dmn['d'])
    out_dan = SDT(DATA_DPRIME['HIT_state2'][numsub_i],DATA_DPRIME['MISS_state2'][numsub_i],DATA_DPRIME['FA_state2'][numsub_i],DATA_DPRIME['CR_state2'][numsub_i])
    dprime2.append(out_dan['d'])

NEWDATA_DPRIME = pd.DataFrame()
NEWDATA_DPRIME['State 1'] = dprime1
NEWDATA_DPRIME['State 2'] = dprime2
NEWDATA_DPRIME.index = subs
DATA_scat_dprime = pd.melt(NEWDATA_DPRIME)
plt.subplot(1,3,3)
ax = sns.barplot(y=np.mean(NEWDATA_DPRIME),x=np.arange(2), data=DATA_scat_dprime, palette=MyPalette)
sns.stripplot(x='variable',y='value',data=DATA_scat_dprime,jitter=0,color='black',alpha=0.3)
plt.plot([0,1],NEWDATA_DPRIME.T,color='black',alpha=0.3)
ax.set_ylim(1.5,5)
ax.set_ylabel("d prime", fontsize=15) #title
ax.set_xlabel("") #title
if save_flag==1:plt.savefig(fig_dir + '/Summary_PerformanceAnd_all.pdf')
if save_flag==1:plt.savefig(fig_dir + '/Summary_PerformanceAnd_all.png')


# print(stats.wilcoxon(DATA_scat_VTC.query('variable=="State 1"')["value"],DATA_scat_VTC.query('variable=="State 2"')["value"]))
# print(stats.wilcoxon(DATA_scat_RT.query('variable=="State 1"')["value"],DATA_scat_RT.query('variable=="State 2"')["value"]))
# print(stats.wilcoxon(DATA_scat_dprime.query('variable=="State 1"')["value"],DATA_scat_dprime.query('variable=="State 2"')["value"]))

d_VTC = CalculateEffect(DATA_scat_VTC.query('variable=="State 1"')["value"].reset_index(drop=True),DATA_scat_VTC.query('variable=="State 2"')["value"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat_VTC.query('variable=="State 1"')["value"],DATA_scat_VTC.query('variable=="State 2"')["value"]),d_VTC)
pg.compute_effsize(DATA_scat_VTC.query('variable=="State 1"')["value"].reset_index(drop=True),DATA_scat_VTC.query('variable=="State 2"')["value"].reset_index(drop=True))

d_RT = CalculateEffect(DATA_scat_RT.query('variable=="State 1"')["value"].reset_index(drop=True),DATA_scat_RT.query('variable=="State 2"')["value"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat_RT.query('variable=="State 1"')["value"],DATA_scat_RT.query('variable=="State 2"')["value"]),d_RT)

d_dprime = CalculateEffect(DATA_scat_dprime.query('variable=="State 1"')["value"].reset_index(drop=True),DATA_scat_dprime.query('variable=="State 2"')["value"].reset_index(drop=True))
print(stats.ttest_rel(DATA_scat_dprime.query('variable=="State 1"')["value"],DATA_scat_dprime.query('variable=="State 2"')["value"]),d_dprime)



tmp = DATA.query('summary_state==0').groupby(['subid','session'])['CorrectOmission'].value_counts().unstack()[1].fillna(0).reset_index()
DATA_DPRIME_DMN = pd.DataFrame()
DATA_DPRIME_DMN['subid'] = tmp.subid
DATA_DPRIME_DMN['session'] = tmp.session
DATA_DPRIME_DMN['HIT'] = DATA.query('summary_state==0').groupby(['subid','session'])['CorrectOmission'].value_counts().unstack()[1].fillna(0).reset_index()[1]
DATA_DPRIME_DMN['MISS'] = DATA.query('summary_state==0').groupby(['subid','session'])['CommissionError'].value_counts().unstack()[1].fillna(0).reset_index()[1]
DATA_DPRIME_DMN['CR'] = DATA.query('summary_state==0').groupby(['subid','session'])['CorrectCommission'].value_counts().unstack()[1].fillna(0).reset_index()[1]
DATA_DPRIME_DMN['FA'] = DATA.query('summary_state==0').groupby(['subid','session'])['OmissionError'].value_counts().unstack()[1].fillna(0).reset_index()[1]
DATA_DPRIME_DMN['VTC'] = DATA.query('summary_state==0').groupby(['subid','session'])['VarianceTimeCourse'].mean().reset_index()['VarianceTimeCourse']
DATA_DPRIME_DMN['state'] = 'DMN'
DATA_DPRIME_DAN = pd.DataFrame()
DATA_DPRIME_DAN['subid'] = tmp.subid
DATA_DPRIME_DAN['session'] = tmp.session
DATA_DPRIME_DAN['HIT'] = DATA.query('summary_state==1').groupby(['subid','session'])['CorrectOmission'].value_counts().unstack()[1].fillna(0).reset_index()[1]
DATA_DPRIME_DAN['MISS'] = DATA.query('summary_state==1').groupby(['subid','session'])['CommissionError'].value_counts().unstack()[1].fillna(0).reset_index()[1]
DATA_DPRIME_DAN['CR'] = DATA.query('summary_state==1').groupby(['subid','session'])['CorrectCommission'].value_counts().unstack()[1].fillna(0).reset_index()[1]
DATA_DPRIME_DAN['FA'] = DATA.query('summary_state==1').groupby(['subid','session'])['OmissionError'].value_counts().unstack()[1].fillna(0).reset_index()[1]
DATA_DPRIME_DAN['VTC'] = DATA.query('summary_state==1').groupby(['subid','session'])['VarianceTimeCourse'].mean().reset_index()['VarianceTimeCourse']
DATA_DPRIME_DAN['state'] = 'DAN'

DATA_DPRIME = pd.concat([DATA_DPRIME_DMN,DATA_DPRIME_DAN]).reset_index(drop=True)

dprime = []
for numsub_i in range(len(DATA_DPRIME)):
    out = SDT(DATA_DPRIME['HIT'][numsub_i],DATA_DPRIME['MISS'][numsub_i],DATA_DPRIME['FA'][numsub_i],DATA_DPRIME['CR'][numsub_i])
    dprime.append(out['d'])

DATA_DPRIME['dprime']=dprime

# import statsmodels.api as sm
# sm.MixedLM.from_formula("dprime ~ state", DATA_DPRIME, groups=DATA_DPRIME["subid"]).fit().summary()
# sm.MixedLM.from_formula("VTC ~ state", DATA_DPRIME, groups=DATA_DPRIME["subid"]).fit().summary()

