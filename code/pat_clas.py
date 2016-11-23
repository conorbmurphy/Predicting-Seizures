
from os import listdir
import numpy as np
from scipy.io import loadmat
import pandas as pd
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import multiprocessing
from scipy import signal
from scipy.stats import kurtosis, skew, pearsonr
from itertools import combinations

class Patient(object):
    def __init__(self, patient_number, path, train=True):
        self.patient_number = int(patient_number)
        self.path = path + '/'
        if train:
            self.files = self.list_files()
        else:
            self.files = self.list_files_test()
        self.reduced = None

    def list_files(self):
        '''
        INPUT: None
        OUTPUT: None, updates file list in ascending order
        '''
        files = []
        for file in listdir(self.path):
            files.append(file.replace('.', '_').split('_')[1:3])
        files = sorted(files, key=lambda num: int(num[0]))
        return [str(self.patient_number)+'_'+file[0]+'_'+file[1]+'.mat' for file in files]

    def list_files_test(self):
        '''
        INPUT: None
        OUTPUT: None, updates file list in ascending order
        '''
        files = []
        for file in listdir(self.path):
            files.append(file.replace('.', '_').split('_')[2])
        files = sorted(files, key=lambda num: int(num))
        return ['new_'+str(self.patient_number)+'_'+file+'.mat' for file in files]

    def return_labels(self):
        '''
        INPUT: none
        OUTPUT: numpy array of recording number and class
        '''
        result = [[int(j[0]), int(j[1][0])] for j in [i.split('_')[1:] for i in self.files]]
        return np.sort(np.array(result))


    def fit_train(self, function, verbose=True):
        '''
        INPUT: none
        OUTPUT: none
        Fits training data using the path
        '''
        l, c = len(self.files), 0
        if self.patient_number == 1:
            self.files.remove('1_45_1.mat') # remove corrupted  '1_45_1.mat', which is file 641
            # files = self.files[642:].append(files)
        for file in self.files:
            path = self.path + file
            new_file = loadmat(path)['dataStruct'][0][0][0]
            i, clas = file.replace('.', '_').split('_')[1:3]
            if self.reduced != None:
                self.reduced = np.append(self.reduced, function(i, clas, new_file), axis=0)
            else:
                self.reduced = function(i, clas, new_file)
            if verbose:
                print 'Fitting file {} / {} complete: file {}'.format(c, l, file)
            c += 1
        if verbose:
            print "Fit all {} files successfully (skipped one of paitent A's files)".format(c)

    def fit_test(self, function, verbose=True):
        '''
        INPUT: none
        OUTPUT: none
        Fits test data using the path
        '''
        l, c = len(self.files), 0
        for file in self.files:
            path = self.path + file
            new_file = loadmat(path)['dataStruct'][0][0][0]
            i = file.replace('.', '_').split('_')[2]
            if self.reduced != None:
                self.reduced = np.append(self.reduced, function(i, None, new_file), axis=0)
            else:
                self.reduced = function(i, None, new_file)
            if verbose:
                print 'Fitting file {} / {} complete: file {}'.format(c, l, file)
            c += 1
        if verbose:
            print "Fit all {} files successfully".format(c)


    def _reduce_16_1000(self, i, clas, mat):
        '''
        INPUT: matrix
        OUTPUT: returns patient information reduced to 1 channel averaged every 1000 inputs
        if there are all zero values, it will use the value from the most recent window
        '''
        temp_mat = mat.mean(axis=1)
        start = 0
        result = []
        for segment in xrange(1000, temp_mat.shape[0]+1000, 1000):
            subset = temp_mat[start:segment]
            subset = subset[subset != 0]
            if len(subset) > 0:
                result.append(subset.mean())
            elif len(result) > 0:
                result.append(result[-1])
            else:
                result.append(0)
            start = segment+1
        if clas:
            result.extend([i, clas])
        else:
            result.extend([i])
        return np.array(result).reshape(1,-1)

    def _reduce_simple_channel_avg(self, i, clas, mat):
        '''
        INPUT: matrix
        OUTPUT: returns each channel average
        '''
        temp_mat = list(mat.mean(axis=0))
        if clas:
            temp_mat.extend([i, clas])
        else:
            temp_mat.extend([i])
        return np.array(temp_mat).reshape(1,-1)

    def _reduce_complex_channel_avg(self, i, clas, mat):
        '''
        INPUT: matrix
        OUTPUT: returns patient information reduced to 16 channels averaged every 24000 inputs
        if there are all zero values, it will use the value from the most recent window
        '''
        temp_mat = np.copy(mat)
        start = 0
        result = np.array([])
        for segment in xrange(24000, 264000, 24000):
            subset = temp_mat[start:segment,:]
            subset = subset[np.all(subset, axis=1)]
            if len(subset) > 0:
                result = np.append(result, subset.mean(axis=0))
            elif len(result) > 0:
                result = np.append(result, result[-16:])
            else:
                result = np.append(result, np.zeros(16))
            start = segment+1
        if clas:
            result = np.append(result, np.array([i, clas]))
        else:
            result = np.append(result, i)
        return result.flatten().reshape(1,-1)


        temp_mat = list(mat.mean(axis=0))
        if clas:
            temp_mat.extend([i, clas])
        else:
            temp_mat.extend([i])
        return np.array(temp_mat).reshape(1,-1)

    def _save_reduce(self, name):
        '''
        INPUT: name of file to be saved, with path if needed
        OUTPUT: None. Saves file
        '''
        df = pd.DataFrame(self.reduced) # done with pandas due to formatting issues
        df.to_csv(name, index=False)

    def print_stats(self):
        '''
        INPUT: None
        OUTPUT: Basic summar stats on three patients
        '''
        print 'Patient A has {} of {} segments as class 1 ({} positive)'.format(150, 1301, float(150)/1301)
        print 'Patient B has {} of {} segments as class 1 ({} positive)'.format(150, 2346, float(150)/2346)
        print 'Patient C has {} of {} segments as class 1 ({} positive)'.format(150, 2394, float(150)/2394)
    # 
    # def plot_file(self, file=None, sixteen_channel=True):
    #     '''
    #     INPUT: none, optional choice to print one condensed plot
    #     OUTPUT: plot of random value
    #     '''
    #     path = 'data/train_' + str(self.patient_number) + '/'
    #     if not x:
    #         file = np.random.choice(self.files)
    #     data = loadmat(path+file)['dataStruct'][0][0][0]
    #     if int(file[-5]) == 1:
    #         c = 'r'
    #     else:
    #         c = 'b'
    #     if sixteen_channel:
    #         fig, ax_list = plt.subplots(16,1)
    #         for ax, flips in zip(ax_list.flatten(), data.transpose()):
    #             ax.plot(flips, c=c)
    #     else:
    #         plt.plot(data, c=c)
    #     plt.show()

def reduce_series(t):
    '''
    INPUT: tuple of patient specs, test or train, and dest file
    OUTPUT: none
    '''
    patient_specs, train_or_test, dest_file = t[0], t[1], t[2]
    if train_or_test == 'train':
        a = Patient(patient_specs[0], patient_specs[1])
        a.fit_train(a._reduce_complex_channel_avg)
        a._save_reduce(dest_file)
    elif train_or_test == 'test':
        b = Patient(patient_specs[0], patient_specs[1], train=False)
        b.fit_test(b._reduce_complex_channel_avg)
        b._save_reduce(dest_file)
    else:
        print 'Problem with train_or_test'
    print 'Saved file {}'.format(dest_file)

def reduce_parallel():
    '''
    INPUT: paths to the test data, name of destination file
    OUTPUT: None
    example:
        paths = ['data/train_1', 'data/train_2', 'data/train_3']
        destinations = ['data/a_reduced2.csv', 'data/b_reduced2.csv', 'data/c_reduced2.csv']
        reduce_parallel(paths, destinations)
    '''
    params = [(('1', '/data/train_1'), 'train', 'data/a_reduced5.csv'),
            (('2', '/data/train_2'), 'train', 'data/b_reduced5.csv'),
            (('3', '/data/train_3'), 'train', 'data/c_reduced5.csv'),
            (('1', '/data/test_1_new'), 'test', 'data/a_test_reduced5.csv'),
            (('2', '/data/test_2_new'), 'test', 'data/b_test_reduced5.csv'),
            (('3', '/data/test_3_new'), 'test', 'data/c_test_reduced5.csv')]
    pool = multiprocessing.Pool(40)
    output = pool.map(reduce_series, params[:2])

def create_reduced_4():
    '''
    Creates a fourth reduced file from 3 by dropping file name and 0's
    '''
    a_df = pd.read_csv('data/a_reduced13.csv').drop('618', axis=1)
    b_df = pd.read_csv('data/b_reduced13.csv').drop('618', axis=1)
    c_df = pd.read_csv('data/c_reduced13.csv').drop('618', axis=1)

    result = []
    for row in a_df.iterrows():
        result.append(all(row[1][:-1] == 0))
    result[:] = [not x for x in result]
    a_df[result].to_csv('data/a_reduced14.csv', index=False) # drops 34 files

    result = []
    for row in b_df.iterrows():
        result.append(all(row[1][:-1] == 0))
    result[:] = [not x for x in result]
    b_df[result].to_csv('data/b_reduced14.csv', index=False) # drops 32 files

    result = []
    for row in c_df.iterrows():
        result.append(all(row[1][:-1] == 0))
    result[:] = [not x for x in result]
    c_df[result].to_csv('data/c_reduced14.csv', index=False) # drops 5 files

    print 'Create reduced training files complete'

def reduce_one_file(file_name):
    '''
    Reuces one test or train recording by file name
    '''
    print 'Fitting {}'.format(file_name)
    temp_mat = loadmat(file_name)['dataStruct'][0][0][0]
    patient, id, clas = file_name.replace('.', '_').replace('/','_').split('_')[4:7]
    if patient == 'new': # in case of test data
         patient, id = file_name.replace('.', '_').replace('/','_').split('_')[6:8]
	 clas = None

    start = 0
    result = np.array([])
    for segment in xrange(24000, 264000, 24000):
        subset = temp_mat[start:segment,:]
        subset = subset[np.all(subset, axis=1)]
        if len(subset) > 0:
            result = np.append(result, subset.mean(axis=0))
        elif len(result) > 0:
            result = np.append(result, result[-16:])
        else:
            result = np.append(result, np.zeros(16))
        start = segment+1

    result2 = np.array([])
    for i in range(16):
        cwtavg = signal.cwt(temp_mat[:,i], signal.ricker, np.linspace(50, 500, 20, endpoint=False))
        result2 = np.concatenate([result2, (cwtavg**2).mean(axis=1)])

    result = np.append(result, result2)

    result = np.append(result, kurtosis(temp_mat)) # adds kurtosis for each channel 
    result = np.append(result, skew(temp_mat)) # adds skew for each channel

    temp_mat2 = temp_mat[np.all(temp_mat, axis=1)]
    if temp_mat2.shape[0] > 0:
        arith_mean_channel = temp_mat2.mean(axis=0)
        arith_mean_total = temp_mat2.mean()
        variance_channel = temp_mat2.var(axis=0, ddof=1)
        variance_total = temp_mat2.var(ddof=1)
        variance_of_variances = variance_channel.var(ddof=1) # this is very high
        std_channel = temp_mat2.std(axis=0)
        std_total = temp_mat2.std()
        max_channel = temp_mat2.max(axis=0)
        max_total = temp_mat2.max()
        min_channel = temp_mat2.min(axis=0)
        min_total = temp_mat2.min()
        median_channel = np.median(temp_mat2, axis=0)
        median_total = np.median(temp_mat2)

        correlations = []
        for c in combinations(range(16), 2):
            correlations.append(pearsonr(temp_mat2[c[0]], temp_mat2[c[1]])[0])
        corr_mean = np.mean(correlations)
        corr_var = np.var(correlations)

        result_mom = np.hstack(np.array([arith_mean_channel,
                        arith_mean_total,
                        variance_channel,
                        variance_total,
                        variance_of_variances,
                        std_channel,
                        std_total,
                        max_channel,
                        max_total,
                        min_channel,
                        min_total,
                        median_channel,
                        median_total,
                        corr_mean,
                        corr_var]).flat).reshape(1,-1)
    else:
        result_mom = np.zeros(105).reshape(1, 105)

    result = np.append(result, result_mom)

    if clas:
        result = np.append(result, np.array([patient, id, clas]))
    else:
        result = np.append(result, np.array([patient, id]))
    return result.flatten().reshape(1,-1)


def reduce_parallel2():
    '''
    to run on aws, change pool and paths
    '''
    params = [('1', '/data/train_1', 'train', 'data/a_reduced13.csv'),
            ('2', '/data/train_2', 'train', 'data/b_reduced13.csv'),
            ('3', '/data/train_3', 'train', 'data/c_reduced13.csv'),
            ('1', '/data/test_1_new', 'test', 'data/a_test_reduced13.csv'),
            ('2', '/data/test_2_new', 'test', 'data/b_test_reduced13.csv'),
            ('3', '/data/test_3_new', 'test', 'data/c_test_reduced13.csv')]
    files = [listdir(param[1]) for param in params]
    files[0].remove('1_45_1.mat') # removes corrupt file
    paths = [param[1] for param in params]
    dest_file = [param[3] for param in params]
    outputs = []
    for i in range(6):
        dfile = dest_file[i]
        ifile = [paths[i]+'/'+file for file in files[i]]
        print 'Launching pool to construct {}'.format(dfile)
        pool = multiprocessing.Pool(40)
        output = pool.map(reduce_one_file, ifile)
        pd.DataFrame(np.concatenate(output)).to_csv(dfile, index=False)
        print 'Saved file {}'.format(dfile)

def pull_recording_num(file):
        mat = loadmat('/data/'+file)['dataStruct'][0][0][4][0]
        return (file.replace('/','.').split('.')[-2], int(mat))

def pull_recording_num_pool():
    result = []
    for directory in ['train_1', 'train_2', 'train_3']:
        file_list = [directory+'/'+i for i in listdir('/data/'+directory)]
        if directory == 'train_1':
            file_list.remove(directory+'/'+'1_45_1.mat')
        pool = multiprocessing.Pool(40)
        output = pool.map(pull_recording_num, file_list)
        result.append(output)
    result = [item for sublist in result for item in sublist]
    pd.DataFrame(result).to_csv('data/recording_numbers.csv', index=False)
    print 'Completed pulling recording numbers and saved to data/recording_numbers.csv'

if __name__ == '__main__':
    # a = Patient(1, 'data/train_1')
    # a.fit_train(a._reduce_complex_channel_avg)
    # a._save_reduce('data/a_reduced3.csv')
    #
    # # a_df = pd.read_csv('data/a_reduced.csv').iloc[:,1:]
    #
    # b = Patient(2, 'data/train_2')
    # b.fit_train(b._reduce_complex_channel_avg)
    # b._save_reduce('data/b_reduced3.csv')
    #
    # # # b_df = pd.read_csv('data/b_reduced.csv').iloc[:,1:]
    #
    # c = Patient(3, 'data/train_3')
    # c.fit_train(c._reduce_complex_channel_avg)
    # c._save_reduce('data/c_reduced3.csv')
    #
    # # # c_df = pd.read_csv('data/c_reduced.csv').iloc[:1:]
    #
    # a_test = Patient(1, 'data/test_1_new', train=False)
    # a_test.fit_test(a_test._reduce_complex_channel_avg)
    # a_test._save_reduce('data/a_test_reduced3.csv')
    #
    # b_test = Patient(2, 'data/test_2_new', train=False)
    # b_test.fit_test(b_test._reduce_complex_channel_avg)
    # b_test._save_reduce('data/b_test_reduced3.csv')
    #
    # c_test = Patient(3, 'data/test_3_new', train=False)
    # c_test.fit_test(c_test._reduce_complex_channel_avg)
    # c_test._save_reduce('data/c_test_reduced3.csv')
    # reduce_parallel()
    # 
    # reduce_parallel2()
    # create_reduced_4() # removes zeros
    pull_recording_num_pool()
