### CHANGE / IN COMPILE_FILES

from scipy.io import loadmat
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
import numpy as np
import seaborn as sns

def file_names(patient_no):
    '''
    INPUT: Patient number
    OUTPUT: two lists of interictal and preictal file names
    '''
    interictal = []
    preictal = []
    for segment in range(13,19):
        interictal.append('1_{}_0.mat'.format(segment))
        preictal.append('1_{}_1.mat'.format(segment))
    return interictal, preictal

def compile_files(interictal, preictal):
    '''
    INPUT: lists of file names for interictal and preictal recordings
    OUTPUT: compiled recordings across segments
    '''
    file = '../data/train_'+interictal[0][0]+'/'
    i_files = []
    p_files = []
    for i_file, p_file in zip(interictal, preictal):
        i_files.append(loadmat(file+i_file)['dataStruct'][0][0][0])
        p_files.append(loadmat(file+p_file)['dataStruct'][0][0][0])
    i_files = np.array([item for subfile in i_files for item in subfile])
    p_files = np.array([item for subfile in p_files for item in subfile])
    return i_files, p_files

def plot_segments(segment, title, color, name):
    '''
    INPUT: numpy array - combined 16-channel recordings
        str - a color
        str - destination file
    OUTPUT: saves figure as name
    '''
    fig, ax_list = plt.subplots(16,1, figsize=(20, 10))
    for ax, flips in zip(ax_list.flatten(), segment.transpose()):
        ax.plot(flips, c=color)
        ax.set_yticks([])
        ax.set_xticks([])
    plt.suptitle(title)
    plt.savefig(name)

def plot_kde(interictal_sample, preictal_sample, title, name):
    '''
    INPUT: 1D numpy arrays of interictal and preictal samples
    OUTPUT: saves figure as name
    '''
    plt.figure(figsize=(20, 10))
    kde = sns.kdeplot(interictal_sample, shade=True, color="b")
    kde = sns.kdeplot(preictal_sample, shade=True, color="r")
    kde.set(xlim=(-65,65))
    kde.set_title(title)
    # sns.plt.xlim(-65,65)
    # sns.plt.title(title)
    plt.savefig(name)

def plot_channel_kde(interictal_sample, preictal_sample, title, name):
    '''
    INPUT: 2D numpy arrays of interictal and preictal samples
    OUTPUT: kde plots across channels
    '''
    fig, ax_list = plt.subplots(4, 4, figsize=(10, 10))
    for ax, flips in zip(ax_list.flatten(), range(16)):
        ax.sns.kdeplot(interictal_sample[:,flips], shade=True, color="b")
        ax.sns.kdeplot(preictal_sample[:,flips], shade=True, color="r")
    plt.suptitle(title)
    plt.savefig(name)

if __name__ == '__main__':
    interictal, preictal = file_names(1)
    i_compiled, p_compiled = compile_files(interictal, preictal)

    plot_segments(i_compiled,
        'One Hour Interictal (Baseline) Recording',
        'b',
        'figures/interictal.png')
    plot_segments(p_compiled,
        'One Hour Preictal (pre-seizure) Recording',
        'r',
        'figures/preictal.png')

    plot_kde(i_compiled.flatten(),
        p_compiled.flatten(),
        'Kernel Density Plot of One Hour Recording Pre- and Interictal',
        'figures/kde.png')

    plot_kde(i_compiled,
        p_compiled,
        'Kernel Density Plots by Channel Pre- and Interictal',
        'figures/kde2.png')
