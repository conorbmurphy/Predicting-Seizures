### CHANGE / IN COMPILE_FILES

from scipy.io import loadmat
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
import numpy as np

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
    plt.title(title)
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
