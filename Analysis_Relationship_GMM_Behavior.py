# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Check relationship between behavior and GMM state

from IPython.display import HTML
HTML("""
<button id="code-show-switch-btn">Hide script</button>

<script>
var code_show = true;

function switch_display_setting() {
    var switch_btn = $("#code-show-switch-btn");
    if (code_show){
    $("div.input").hide();
    code_show = false;
    switch_btn.text("Show script");
    }
    else {
    $("div.input").show();
    code_show = true;
    switch_btn.text("Hide script");
    }

}

$("#code-show-switch-btn").click(switch_display_setting)
""")

import glob
import pandas as pd
import numpy as np
from numpy import matlib as mb
from matplotlib import pyplot as plt
import seaborn as sns
import os
from scipy import stats
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
from scipy import io
from sklearn import datasets
from sklearn import mixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
import math
import pickle
import statsmodels.stats.anova as anova
import scipy.cluster.hierarchy as shc
import scipy.io

Z = norm.ppf


# +
def CheckWhere(num_vol,tr,onset_time):
    onset_time = onset_time + 5 # explain for hemodynamic response
    if onset_time>num_vol*tr: onset_time = num_vol*tr-1 
    x = np.arange(tr, round((num_vol+1)*tr,2),tr)
    belong = np.logical_and(onset_time <= x,onset_time>x-tr)
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

def MakeData(DATA,DMN_state,DAN_state,num_state):
    tmp = np.arange(num_state)+1
    NEWVTCDATA = pd.DataFrame()
    NEWVTCDATA['DMN_state'] = DATA.query('state==%s' %list(tmp[DMN_state])).groupby(['subid'])['VarianceTimeCourse'].mean()
    NEWVTCDATA['DAN_state'] = DATA.query('state==%s' %list(tmp[DAN_state])).groupby(['subid'])['VarianceTimeCourse'].mean()
    DATA_scat_vtc = pd.melt(NEWVTCDATA)

    ## comparing of RT
    NEWDATA_RT = pd.DataFrame()
    NEWDATA_RT['DMN_state'] = DATA.query('state==%s' %list(tmp[DMN_state])).groupby(['subid'])['ReactionTime_Interpolated'].mean()
    NEWDATA_RT['DAN_state'] = DATA.query('state==%s' %list(tmp[DAN_state])).groupby(['subid'])['ReactionTime_Interpolated'].mean()
    DATA_scat_RT = pd.melt(NEWDATA_RT)

    ## comparing of MW
    NEWDATA_MW = pd.DataFrame()
    NEWDATA_MW['DMN_state'] = DATA.query('state==%s' %list(tmp[DMN_state])).groupby(['subid'])['ThoughtProbe_Interpolated'].mean()
    NEWDATA_MW['DAN_state'] = DATA.query('state==%s' %list(tmp[DAN_state])).groupby(['subid'])['ThoughtProbe_Interpolated'].mean()
    DATA_scat_MW = pd.melt(NEWDATA_MW)


    DATA_DPRIME = pd.DataFrame()
    DATA_DPRIME['HIT_DMN'] = DATA.query('state==%s' %list(tmp[DMN_state])).groupby(['subid'])['CorrectOmission'].value_counts().unstack()[1].fillna(0)
    DATA_DPRIME['MISS_DMN'] = DATA.query('state==%s' %list(tmp[DMN_state])).groupby(['subid'])['CommissionError'].value_counts().unstack()[1].fillna(0)
    DATA_DPRIME['CR_DMN'] = DATA.query('state==%s' %list(tmp[DMN_state])).groupby(['subid'])['CorrectCommission'].value_counts().unstack()[1].fillna(0)
    DATA_DPRIME['FA_DMN'] = DATA.query('state==%s' %list(tmp[DMN_state])).groupby(['subid'])['OmissionError'].value_counts().unstack()[1].fillna(0)
    DATA_DPRIME['HIT_DAN'] = DATA.query('state==%s' %list(tmp[DAN_state])).groupby(['subid'])['CorrectOmission'].value_counts().unstack()[1].fillna(0)
    DATA_DPRIME['MISS_DAN'] = DATA.query('state==%s' %list(tmp[DAN_state])).groupby(['subid'])['CommissionError'].value_counts().unstack()[1].fillna(0)
    DATA_DPRIME['CR_DAN'] = DATA.query('state==%s' %list(tmp[DAN_state])).groupby(['subid'])['CorrectCommission'].value_counts().unstack()[1].fillna(0)
    DATA_DPRIME['FA_DAN'] = DATA.query('state==%s' %list(tmp[DAN_state])).groupby(['subid'])['OmissionError'].value_counts().unstack()[1].fillna(0)
    
    dprime_DMN = []
    dprime_DAN = []
    for numsub_i, sub_i in enumerate(pd.unique(DATA.subid)):
        out_dmn = SDT(DATA_DPRIME['HIT_DMN'][numsub_i],DATA_DPRIME['MISS_DMN'][numsub_i],DATA_DPRIME['FA_DMN'][numsub_i],DATA_DPRIME['CR_DMN'][numsub_i])
        dprime_DMN.append(out_dmn['d'])
        out_dan = SDT(DATA_DPRIME['HIT_DAN'][numsub_i],DATA_DPRIME['MISS_DAN'][numsub_i],DATA_DPRIME['FA_DAN'][numsub_i],DATA_DPRIME['CR_DAN'][numsub_i])
        dprime_DAN.append(out_dan['d'])
    
    NEWDATA_DPRIME = pd.DataFrame()
    NEWDATA_DPRIME['DMN_state'] = dprime_DMN
    NEWDATA_DPRIME['DAN_state'] = dprime_DAN
    DATA_scat_dprime = pd.melt(NEWDATA_DPRIME)
    return DATA_scat_vtc, DATA_scat_RT, DATA_scat_dprime, DATA_scat_MW

# +
##############
# parameters #
##############
save_flag = 1

# task = 'Original'
# task = 'Rest'
task = "Replication"
# task = "ADHD"
# task = "Reward"
# task = "nonReward"

top_dir = 'C:/Users/ayumu/Dropbox/gradCPT/'
fig_dir = top_dir + 'fig/NeuroImage/'
performance_list = ['CommissionError', 'CorrectCommission', 'CorrectOmission', 'OmissionError', 'In_the_Zone', 'VarianceTimeCourse','ReactionTime_Interpolated']

## Schaefer400_7Net
roi = 'Schaefer400_7Net'
roi_dir = 'C:/Users/ayumu/Dropbox/gradCPT/Parcellations/' + roi + '/'
ROI_files = pd.read_csv(roi_dir + 'Schaefer400_7Net.csv')
roiname = ROI_files.Network
network = np.unique(roiname)

# demographic data
if task == 'Original':
    project = 'OriginalGradCPT'
    demo = pd.read_csv(glob.glob(top_dir + '/code/participants.tsv')[0],delimiter='\t')
    basin_dir = top_dir + 'data/%s/EnergyLandscape/' %project
    mat = scipy.io.loadmat(basin_dir + roi + '/LocalMin_Summary.mat')
    tr = 2.0
elif task == 'Rest':
    project = 'Rest'
    demo = pd.read_csv(glob.glob(top_dir + '/code/participants.tsv')[0],delimiter='\t')
    basin_dir = top_dir + 'data/%s/EnergyLandscape/' %project
    mat = scipy.io.loadmat(basin_dir + roi + '/LocalMin_Summary.mat')
    tr = 2.0
elif task == 'Replication':
    project = 'GradCPT_MindWandering'
    demo = pd.read_csv(glob.glob(top_dir + '/code/participants_HC.tsv')[0],delimiter='\t')
    basin_dir = top_dir + 'data/%s/EnergyLandscape/' %project
    mat = scipy.io.loadmat(basin_dir + roi + '/HC_onlyMW/LocalMin_Summary.mat')
    tr=1.08
elif task == 'ADHD':
    project = 'GradCPT_MindWandering'
    demo = pd.read_csv(glob.glob(top_dir + '/code/participants_ADHD.tsv')[0],delimiter='\t')
    basin_dir = top_dir + 'data/%s/EnergyLandscape/' %project
    mat = scipy.io.loadmat(basin_dir + roi + '/ADHD_onlyMW/LocalMin_Summary.mat')
    tr=1.08
elif task == 'Reward':
    project = 'GradCPT_reward'
    demo = pd.read_csv(glob.glob(top_dir + '/code/participants_reward.tsv')[0],delimiter='\t')
    basin_dir = top_dir + 'data/%s/EnergyLandscape/' %project
    mat = scipy.io.loadmat(basin_dir + roi + '/reward/LocalMin_Summary.mat')
    tr = 2.0
elif task == 'nonReward':
    project = 'GradCPT_reward'
    demo = pd.read_csv(glob.glob(top_dir + '/code/participants_reward.tsv')[0],delimiter='\t')
    basin_dir = top_dir + 'data/%s/EnergyLandscape/' %project
    mat = scipy.io.loadmat(basin_dir + roi + '/nonreward/LocalMin_Summary.mat')
    tr = 2.0

source_dir = top_dir + 'data/%s/MRI/' %project 
out_dir = '%s/data/%s/GMM/' %(top_dir,project)
if os.path.isdir(out_dir)==False: os.mkdir(out_dir)

subs = demo.participants_id
sub_num = len(subs)


# load data
max_iter = 100000
n_init = 10
with open(out_dir + "GMM%s_itr%s_init%s_task_%s.pkl" % (20,max_iter,n_init,task), "rb") as file: dpgmm = pickle.load(file)   


data = pd.DataFrame()
for sub_i in subs:
    data_file = glob.glob(top_dir + 'data/%s/time_series/%s/NetworkROI/*%s*.dat' % (project, roi,sub_i));data_file.sort()
    for i, file_i in enumerate(data_file,1):
        data_sub = pd.DataFrame()
        data_tmp = np.genfromtxt(file_i, delimiter=',').T
        data_sub = data_sub.append(pd.DataFrame(dpgmm.predict(data_tmp)))
        data_sub['sub'] = sub_i
        data_sub['run'] = i
        data = data.append(data_sub)
    
df =pd.DataFrame({'timepoint':data.index,
                  'run':data['run'].astype(int),
                  'sub':data['sub'],
                  'state':data[0].values})
df.index = range(len(data))

# +
transition = df['state']
tmp = transition.drop(len(transition)-1)
num_state = 20
use_state = np.unique(df['state'])

trans_matrix = np.zeros([num_state,num_state])
trans_count = np.zeros(num_state)
for time_i,state_i in enumerate(tmp,1):
    trans_matrix[state_i,transition[time_i]] = trans_matrix[state_i,transition[time_i]] + 1
    trans_count[state_i] = trans_count[state_i] + 1
trans_matrix = trans_matrix/mb.repmat(trans_count,num_state,1)
tmp = trans_matrix[use_state]
tmp = tmp.T[use_state]

plt.figure(figsize=(12,6))
plt.subplot(121)
sns.heatmap(tmp, vmin=0, vmax=0.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

Occupancy_rate = np.array([])
for i in range(num_state):
    Occupancy_rate = np.hstack((Occupancy_rate,100*sum(df.state==i)/len(df.state)))
x = np.arange(len(use_state))
plt.subplot(122)
sns.barplot(x=x, y=Occupancy_rate[use_state])   
if save_flag==1: plt.savefig(fig_dir + '/StateTransitionAndPercentage_GMM.png')

# +
# Select a subset of the networks
thleshold = 3 # Use state 2 %
used_networks = np.where(Occupancy_rate > thleshold)
used_columns = dpgmm.means_[used_networks]
df_state = pd.DataFrame(used_columns,columns=network,index=np.round(Occupancy_rate[used_networks]))
sns.clustermap(df_state.T, cmap="vlag",linewidths=.75, row_cluster=False, z_score=1)
if save_flag==1: plt.savefig(fig_dir + '/StableStates_Hierachical_GMM.png')
if save_flag==1: plt.savefig(fig_dir + '/StableStates_Hierachica_GMMl.pdf')
# sns.clustermap(df_state.T, cmap="vlag",linewidths=.75, row_cluster=False, col_cluster=False, z_score=1)

plt.show()    

dend = shc.dendrogram(shc.linkage(normalize(df_state)))
dend['leaves']
# # -

# behavioral performance
DATA = pd.DataFrame()
for sub_i in subs:
    task_files = glob.glob(source_dir + sub_i +'_task-gradCPT*events.tsv');task_files.sort()
    DATA_sub = pd.DataFrame()
    run_run = np.array([])
    state_run = np.array([])
    for num_file_i,task_file_i in enumerate(task_files,1):
        taskinfo = pd.read_csv(task_file_i,delimiter='\t')
        data_tmp = df.query('sub == @sub_i & run == @num_file_i')['state'].reset_index(drop=True)
        num_vol = data_tmp.shape[0]
        for onset_time in taskinfo['onset']:
            belong = CheckWhere(num_vol,tr,onset_time)
            state_run = np.hstack((state_run,data_tmp[belong]))
            run_run = np.hstack((run_run,num_file_i ))   
        DATA_sub = DATA_sub.append(taskinfo.loc[:,performance_list])
        DATA_sub['state'] = state_run
        DATA_sub['run'] = run_run       
        DATA_sub['subid'] = sub_i       
    DATA = DATA.append(DATA_sub)


# +
## figure of Duration
color_list = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628']
STATEDATA = DATA.groupby(['subid'])['state'].value_counts().unstack().fillna(0)
NEWDATA = STATEDATA.apply(lambda x:100*x/sum(x),axis=1)
NEWDATA = NEWDATA[used_networks[0]]
DATA_scat = pd.melt(NEWDATA)
num_state = len(used_networks[0])
# DATA_scat = pd.melt(NEWDATA)
# num_state = STATEDATA.shape[1]

f = plt.figure(figsize=(8,6))
# ax = sns.barplot(y=NEWDATA.mean(),x=np.arange(num_state),facecolor=(1,1,1,0),edgecolor=".2")
ax = sns.barplot(y=NEWDATA.mean().iloc[dend['leaves']],x=np.arange(num_state),edgecolor=".2")
sns.stripplot(x='state',y='value',data=DATA_scat,order=NEWDATA.columns[dend['leaves']],jitter=0,color='black')
ax.set_title("%s State rate" %roi, fontsize=20) #title
ax.set_ylim(0,70)
if save_flag==1: plt.savefig(fig_dir + '/Probability_GMM.png')
if save_flag==1: plt.savefig(fig_dir + '/Probability_GMM.pdf')

# +
AverageState = NEWDATA.mean()
df_average = pd.DataFrame({'state': AverageState.index,
              'value': AverageState.values})
# FirstState = np.where(AverageState == np.sort(AverageState)[-1])[0][0]
# SecondState = np.where(AverageState == np.sort(AverageState)[-2])[0][0]
# TargetState = list(np.where(AverageState > AverageState.drop([FirstState,SecondState]).mean())[0] + 1)
# TargetState = list(np.where(AverageState > AverageState.mean())[0] + 1)
# TargetState = df_average.query('value > @thleshold')['state'].values
# TargetState = np.where(Occupancy_rate>1)[0]
TargetState = np.where(Occupancy_rate>3)[0]

## figure of VTC
VTCDATA = DATA.groupby(['subid','state'])['VarianceTimeCourse'].mean().unstack()
VTCDATA = VTCDATA[TargetState]
DATA_scat = pd.melt(VTCDATA)
f = plt.figure(figsize=(6,6))
ax = sns.barplot(y=VTCDATA.mean().iloc[dend['leaves']],x=np.arange(len(TargetState)), data=DATA_scat)
sns.stripplot(x='state',y='value',data=DATA_scat,order=NEWDATA.columns[dend['leaves']],jitter=0,color='black')
ax.set_ylim(np.min(DATA_scat.value)-0.05,np.max(DATA_scat.value)+0.05)
ax.set_title("Variance time course", fontsize=20) #title
DATA_scat['Subjects'] = np.matlib.repmat(np.arange(len(subs)),1,len(TargetState))[0]
aov=anova.AnovaRM(DATA_scat, 'value','Subjects',['state'])
result=aov.fit()
print(result)
if save_flag==1: plt.savefig(fig_dir + '/VTC_GMM.png')
if save_flag==1: plt.savefig(fig_dir + '/VTC_GMM.pdf')
# -

## figure of RT
NEWDATA_RT = DATA.groupby(['subid','state'])['ReactionTime_Interpolated'].mean().unstack()
NEWDATA_RT = NEWDATA_RT[TargetState]
DATA_scat_RT = pd.melt(NEWDATA_RT)
f = plt.figure(figsize=(6,6))
ax = sns.barplot(y=NEWDATA_RT.mean().iloc[dend['leaves']],x=np.arange(len(TargetState)), data=DATA_scat_RT)
ax.set_title("Reaction time", fontsize=20) #title
sns.stripplot(x='state',y='value',data=DATA_scat_RT,order=NEWDATA.columns[dend['leaves']],jitter=0,color='black')
ax.set_ylim(np.min(DATA_scat_RT.value)-0.05,np.max(DATA_scat_RT.value)+0.05)
DATA_scat_RT['Subjects'] = np.matlib.repmat(np.arange(len(subs)),1,len(TargetState))[0]
aov=anova.AnovaRM(DATA_scat_RT, 'value','Subjects',['state'])
result=aov.fit()
print(result)
if save_flag==1: plt.savefig(fig_dir + '/RT_GMM.png')
if save_flag==1: plt.savefig(fig_dir + '/RT_GMM.pdf')

# +
HIT = DATA.groupby(['subid','state'])['CorrectOmission'].value_counts().unstack()[1].fillna(0).unstack().fillna(0)[TargetState]
MISS = DATA.groupby(['subid','state'])['CommissionError'].value_counts().unstack()[1].fillna(0).unstack().fillna(0)[TargetState]
CR = DATA.groupby(['subid','state'])['CorrectCommission'].value_counts().unstack()[1].fillna(0).unstack().fillna(0)[TargetState]
FA = DATA.groupby(['subid','state'])['OmissionError'].value_counts().unstack()[1].fillna(0).unstack().fillna(0)[TargetState]

NEWDATA_DPRIME = pd.DataFrame()
for state_i in TargetState:
    DATA_state = pd.DataFrame()
    dprime = []
    for sub_i in subs:
        out = SDT(HIT.loc[sub_i][state_i],MISS.loc[sub_i][state_i],FA.loc[sub_i][state_i],CR.loc[sub_i][state_i])
        dprime.append(out['d'])
    DATA_state['dprime'] = dprime
    DATA_state['Subjects'] = subs
    DATA_state['state'] = state_i
    NEWDATA_DPRIME = NEWDATA_DPRIME.append(DATA_state)

f = plt.figure(figsize=(6,6))
ax = sns.barplot(y=NEWDATA_DPRIME.groupby(['state'])['dprime'].mean().iloc[dend['leaves']],x=np.arange(len(TargetState)), data=NEWDATA_DPRIME)
ax.set_title("d prime", fontsize=20) #title
sns.stripplot(x='state',y='dprime',data=NEWDATA_DPRIME,order=NEWDATA.columns[dend['leaves']],jitter=0,color='black')
ax.set_ylim(np.min(NEWDATA_DPRIME.dprime)-0.05,np.max(NEWDATA_DPRIME.dprime)+0.05)
NEWDATA_DPRIME['Subjects'] = np.matlib.repmat(np.arange(len(subs)),1,len(TargetState))[0]
aov=anova.AnovaRM(NEWDATA_DPRIME, 'dprime','Subjects',['state'])
result=aov.fit()
print(result)
if save_flag==1: plt.savefig(fig_dir + '/dprime_GMM.png')
if save_flag==1: plt.savefig(fig_dir + '/dprime_GMM.pdf')


print(stats.ttest_rel(DATA_scat.query("state==6").reset_index(drop=True).value,DATA_scat.query("state==12").reset_index(drop=True).value))
print(stats.ttest_rel(DATA_scat_RT.query("state==6").reset_index(drop=True).value,DATA_scat_RT.query("state==12").reset_index(drop=True).value))
print(stats.ttest_rel(NEWDATA_DPRIME.query("state==6").reset_index(drop=True).dprime,NEWDATA_DPRIME.query("state==12").reset_index(drop=True).dprime))
