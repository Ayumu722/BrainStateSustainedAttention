# -*- coding: utf-8 -*-
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

# # Gaussian Mixture Model for fMRI data

# import tool
import pandas as pd
import numpy as np
import scipy.io
import os
import glob
import pickle
from sklearn import mixture

##############
# parameters #
##############
save_flag = 1

# task = 'Original'
# task = 'Rest'
# task = "Replication"
task = "ADHD"
# task = "Reward"
# task = "nonReward"

top_dir = 'C:/Users/ayumu/Dropbox/gradCPT/'


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
elif task == 'Rest':
    project = 'Rest'
    demo = pd.read_csv(glob.glob(top_dir + '/code/participants.tsv')[0],delimiter='\t')
    basin_dir = top_dir + 'data/%s/EnergyLandscape/' %project
    mat = scipy.io.loadmat(basin_dir + roi + '/LocalMin_Summary.mat')
elif task == 'Replication':
    project = 'GradCPT_MindWandering'
    demo = pd.read_csv(glob.glob(top_dir + '/code/participants_HC.tsv')[0],delimiter='\t')
    basin_dir = top_dir + 'data/%s/EnergyLandscape/' %project
    mat = scipy.io.loadmat(basin_dir + roi + '/HC_onlyMW/LocalMin_Summary.mat')
elif task == 'ADHD':
    project = 'GradCPT_MindWandering'
    demo = pd.read_csv(glob.glob(top_dir + '/code/participants_ADHD.tsv')[0],delimiter='\t')
    basin_dir = top_dir + 'data/%s/EnergyLandscape/' %project
    mat = scipy.io.loadmat(basin_dir + roi + '/ADHD_onlyMW/LocalMin_Summary.mat')
elif task == 'Reward':
    project = 'GradCPT_reward'
    demo = pd.read_csv(glob.glob(top_dir + '/code/participants_reward.tsv')[0],delimiter='\t')
    basin_dir = top_dir + 'data/%s/EnergyLandscape/' %project
    mat = scipy.io.loadmat(basin_dir + roi + '/reward/LocalMin_Summary.mat')
elif task == 'nonReward':
    project = 'GradCPT_reward'
    demo = pd.read_csv(glob.glob(top_dir + '/code/participants_reward.tsv')[0],delimiter='\t')
    basin_dir = top_dir + 'data/%s/EnergyLandscape/' %project
    mat = scipy.io.loadmat(basin_dir + roi + '/nonreward/LocalMin_Summary.mat')


out_dir = '%s/data/%s/GMM/' %(top_dir,project)
if os.path.isdir(out_dir)==False: os.mkdir(out_dir)

subs = demo.participants_id

data = pd.DataFrame()
for sub_i in subs:
    data_file = glob.glob(top_dir + 'data/%s/time_series/%s/NetworkROI/*%s*.dat' % (project, roi,sub_i));data_file.sort()
    for file_i in data_file:
        data_tmp = np.genfromtxt(file_i, delimiter=',').T
        data = data.append(pd.DataFrame(data_tmp))
 
data = pd.DataFrame(data.values,columns=network)
ndim = np.shape(data)

#  Fit Bayesian GMM
max_iter = 100000
n_init = 10
dpgmm = mixture.BayesianGaussianMixture(n_components=20,covariance_type='full',max_iter=max_iter,n_init=n_init).fit(data)
with open(out_dir + "GMM%s_itr%s_init%s_task_%s.pkl" % (20,max_iter,n_init,task),"wb") as file: pickle.dump(dpgmm, file)
