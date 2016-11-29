### CHANGE / IN COMPILE_FILES

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
import numpy as np
#import seaborn as sns
from scipy.io import loadmat
from scipy import signal
import multiprocessing
from code.model import import_data


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
    file = '/data/train_'+interictal[0][0]+'/'
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
    fig, ax_list = plt.subplots(16,1, figsize=(10, 5))
    for ax, flips in zip(ax_list.flatten(), segment.transpose()):
        ax.plot(flips, c=color, linewidth=.5)
        ax.set_yticks([])
        ax.set_xticks([])
    plt.suptitle(title)
    plt.savefig(name)

def plot_kde(interictal_sample, preictal_sample, title, name):
    '''
    INPUT: 1D numpy arrays of interictal and preictal samples
    OUTPUT: saves figure as name
    '''
    plt.figure(figsize=(10, 5))
    kde = sns.kdeplot(interictal_sample, shade=True, color="b")
    kde = sns.kdeplot(preictal_sample, shade=True, color="r")
    kde.set(xlim=(-65,65))
    kde.set_title(title)
    plt.savefig(name)

def plot_channel_kde(interictal_sample, preictal_sample, title, name):
    '''
    INPUT: 2D numpy arrays of interictal and preictal samples
    OUTPUT: kde plots across channels
    '''
    fig, ax_list = plt.subplots(4, 4, figsize=(10, 10))
    for ax, flips in zip(ax_list.flatten(), range(16)):
        kde = sns.kdeplot(interictal_sample[:,flips], shade=True, color="b",
            ax=ax)
        kde = sns.kdeplot(preictal_sample[:,flips], shade=True, color="r",
            ax=ax)
        kde.set(xlim=(-100, 100), ylim=(0, .06))
        kde.set_title('Channel {}'.format(flips+1))
    plt.suptitle(title)
    plt.savefig(name)

def plot_feature_importance(df, name):
    '''
    INPUT: Data frame of feature importances
    OUTPUT: None, saves plot
    '''
    plt.figure(figsize=(10, 5))
    pos = np.arange(6)[::-1]+.5
    plt.barh(pos, df['Value'], align='center', color='#71EEB8')
    plt.yticks(pos, df['Feature'], size='x-small')
    plt.xlabel('Importance', size='smaller')
    plt.suptitle('Feature Importance by Feature Type')
    plt.savefig(name)


def return_frequencies():
    '''
    INPUT: None
    OUTPUT: numpy array of 25 frequencies spaced 5 each over the common
        brain activity wavelengths:
            delta: < 4 hz
            theta: >= 4 hz & < 8 hz
            alpha: >= 8 hz & < 14 hz
            beta:  >= 14 hz & < 32 hz
            gamma: >= 14 hz
    '''
    frequencies = np.concatenate([\
            np.linspace(300, 100, 5), # delta waves
            np.linspace(95, 57, 5), # theta waves
            np.linspace(55, 27, 5), # alpha waves
            np.linspace(25, 13, 5), # beta waves
            np.linspace(12, 2, 5)]) # gamma waves
    return frequencies


def wavelet_spectrogram(mat, title, name):
    '''
    INPUT: iEEG recording as a numpy array, plot title and new file name
    OUTPUT: None, saves plot
    '''
    plt.figure(figsize=(10, 5))
    freq = return_frequencies()
    result = signal.cwt(mat[:,15], signal.ricker, freq)
    plt.imshow(result, extent=[0, 1440000, 2, 300], cmap='PRGn',\
        aspect='auto', vmax=abs(result).max(), vmin=-abs(result).max())
    plt.suptitle(title)
    plt.xlabel('Time (observations sampled at 400 hz)')
    plt.ylabel('Frequency (hz)')
    plt.savefig(name)

def plot_correlations(mat, title, name):
    '''
    INPUT: recording, title, and destination file name
    OUTPUT: None, saves plot
    '''
    plt.figure(figsize=(10, 5))
    sns.heatmap(pd.DataFrame(mat).corr())
    plt.suptitle(title)
    plt.savefig(name)

if __name__ == '__main__':
    interictal, preictal = file_names(1)
    i_compiled, p_compiled = compile_files(interictal, preictal)

    df_concat, test_concat = import_data()

    feature_importances = pd.read_csv('data/feature_importances.csv')

    #plot_segments(i_compiled,
    #    'One Hour Interictal (Baseline) Recording',
    #    'b',
    #    'figures/interictal.png')

    #plot_segments(p_compiled,
    #    'One Hour Preictal (pre-seizure) Recording',
    #    'r',
    #    'figures/preictal.png')

    plot_feature_importance(feature_importances,
        'figures/feature_importance.png')

    #plot_kde(i_compiled.flatten(),
    #    p_compiled.flatten(),
    #    'Kernel Density Plot of One Hour Recording Pre- and Interictal',
    #    'figures/kde.png')

    #plot_channel_kde(i_compiled,
    #    p_compiled,
    #    'Kernel Density Plots by Channel Pre- and Interictal',
    #    'figures/kde2.png')

    #wavelet_spectrogram(i_compiled,
    #    'Interictal Wavelet Spectrogram from Channel 16',
    #    'figures/spectrogram_i.png')

   # wavelet_spectrogram(p_compiled,
   #     'Preictal Wavelet Spectrogram from Channel 16',
   #     'figures/spectrogram_p.png')

   # plot_correlations(i_compiled,
   #     'Interictal Channel Coorelations',
   #     'figures/coorelations_i.png')

   # plot_correlations(p_compiled,
   #     'Precital Channel Coorelations',
   #     'figures/coorelations_p.png')
