
from os import listdir
import numpy as np
from scipy.io import loadmat
import pandas as pd
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import multiprocessing
from scipy import signal
from scipy.stats import kurtosis, skew, pearsonr, entropy, gaussian_kde
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







class Features(object):
    def __init__(self, file_name):
        self.file_name = file_name

        self.temp_mat = None # Full recording
        self.temp_mat2 = None # Non-zero values from temp_mat

        self.patient = None
        self.id = None
        self.clas = None
        self.contaminated = None
        self.sequence = None

        self.means = None
        self.wavelets = None
        self.mom = None
        self.entropies = None
        self.correlations = None

        self.fit()

    def fit(self):
        print 'Fitting {}'.format(self.file_name)
        self.load_file()

        self.metadata()
        self.channel_means()
        self.wavelet_transformation()
        self.method_of_moments()
        self.entropize()
        self.correlate()

        #self.return_results()

    def load_file(self):
        '''
        INPUT: None
        OUTPUT: None
            Loads file and saves to self.temp_mat
            Saves all non-zero values to self.temp_mat2
        '''
        self.temp_mat = loadmat(self.file_name)['dataStruct'][0][0]
        self.temp_mat2 = self.temp_mat[0][np.all(self.temp_mat[0], axis=1)]

    def metadata(self):
        '''
        INPUT: None
        OUTPUT: None, saves the following metadata:
            patient (int, 1-3)
            id (int, recording number)
            clas (int, 0 for interictal, 1 for preictal)
            contaminated (boolean, 0 for not contaminated, 1 for contaminated)
            sequence (int, 1-6, where the recording appears in 1 hour segment-
                for training set only)
        '''
        self.patient, self.id, self.clas = self.file_name.replace('.',\
            '_').replace('/','_').split('_')[4:7]
        self.contaminated = self.file_name.split('/')[-1] in label
        if self.patient == 'new': # in case of test data
             self.patient, self.id = self.file_name.replace('.', '_')\
                .replace('/','_').split('_')[6:8]
             self.clas = None # resets class as none
        else:
        	self.sequence = self.temp_mat[4][0][0] # pulls sequence number (only exists in training set)

    def channel_means(self):
        '''
        INPUT: None
        OUTPUT: None
            Divides the sample into 10 segments and aves a 1x160 numpy array of
            the channel means of all non-zero values to self.means()
        '''
        start = 0
        result = np.array([])
        for segment in xrange(24000, 264000, 24000):
            subset = self.temp_mat[0][start:segment,:]
            subset = subset[np.all(subset, axis=1)]
            if len(subset) > 0:
                result = np.append(result, subset.mean(axis=0))
            elif len(result) > 0:
                result = np.append(result, result[-16:])
            else:
                result = np.append(result, np.zeros(16))
            start = segment+1
        self.means = result.reshape(1,-1)

    def wavelet_transformation(self):
        '''
        INPUT: None
        OUTPUT: None
            Performs a wavelet transformation using 25 freqencies of the
            Ricker wave on each channel, saving the means of the squared result.
            This returns a 1x400 numpy array, 25 values for each channel
        '''
        freq = self.return_frequencies()
        result = np.array([])
        for i in range(16):
            cwtavg = signal.cwt(self.temp_mat2[:,i], signal.ricker, freq)
            result = np.concatenate([result, (cwtavg**2).mean(axis=1)])
        self.wavelets = result.reshape(1,-1)

    def method_of_moments(self):
        '''
        INPUT: None
        OUTPUT: None
            Performs the following method of moments calculations:
                mean by channel - 16 values
                mean for total recording - 1 value
                variance by channel - 16 values
                variance for total recording - 1 value
                variance of channel variances - 1 value
                kurtosis by channel - 16 values
                skew by channel - 16 values
                max of channels - 16 values
                max of total recording - 1 value
                min of channels - 16 values
                min of total recording - 1 value
                median of channels - 16 values
                median of total recording - 1 value
            In case of an all-zeros recording, returns all zeros
        '''
        if self.temp_mat2.shape[0] > 0:
            arith_mean_channel = self.temp_mat2.mean(axis=0)
            arith_mean_total = self.temp_mat2.mean()

            variance_channel = self.temp_mat2.var(axis=0, ddof=1)
            variance_total = self.temp_mat2.var(ddof=1)
            variance_of_variances = variance_channel.var(ddof=1) # this is very high

            kurtosis_channel = kurtosis(self.temp_mat2)
            skew_channel = skew(self.temp_mat2)

            max_channel = self.temp_mat2.max(axis=0)
            max_total = self.temp_mat2.max()
            min_channel = self.temp_mat2.min(axis=0)
            min_total = self.temp_mat2.min()

            median_channel = np.median(self.temp_mat2, axis=0)
            median_total = np.median(self.temp_mat2)

            self.mom = np.hstack(np.array([\
                            arith_mean_channel,
                            arith_mean_total,
                            variance_channel,
                            variance_total,
                            variance_of_variances,
                            kurtosis_channel,
                            skew_channel,
                            max_channel,
                            max_total,
                            min_channel,
                            min_total,
                            median_channel,
                            median_total]).flat).reshape(1,-1)

        else:
            self.mom = np.zeros(118).reshape(1, -1) # BE CAREFUL WITH THIS

    def entropize(self):
        '''
       INPUT: None
        OUTPUT: None
            Saves 16 channel entropies to self.entropies
        '''
        entropies = []
        for col in range(16):
            kde = gaussian_kde(self.temp_mat2[:,col])
            r = np.linspace(min(self.temp_mat2[:,col]),\
                max(self.temp_mat2[:,col]), len(self.temp_mat2[:,col])/1E3)
            entropies.append(entropy(kde.evaluate(r)))
        self.entropies = np.array(entropies).reshape(1,-1)

    def correlate(self):
        '''
        INPUT: None
        OUTPUT: None
            Saves 120 combinations of channel pearson correlations, their mean
            and variance
        '''
        correlations = []
        for c in combinations(range(16), 2):
            correlations.append(pearsonr(self.temp_mat2[c[0]],\
                self.temp_mat2[c[1]])[0])
        corr_mean = np.mean(correlations)
        corr_var = np.var(correlations)
        self.correlations = np.hstack([np.array(correlations), corr_mean, corr_var]).reshape(1,-1)

    def return_results(self):
        '''
        INPUT: None
        OUTPUT: Combined results
        '''
        try:
        	if self.clas:
        		result = np.concatenate([\
                    self.means,
                    self.wavelets,
                    self.mom,
                    self.entropies,
                    self.correlations,
                    np.array([\
                        self.patient,
                        self.id,
                        self.sequence,
                        self.contaminated,
                        self.clas]).reshape(1,-1)], axis=1)
        	else:
                    result = np.concatenate([\
                    self.means,
                    self.wavelets,
                    self.mom,
                    self.entropies,
                    self.correlations,
                    np.array([\
                        self.patient,
                        self.id]).reshape(1,-1)], axis=1)
        	return result.flatten().reshape(1,-1)
        except ValueError:
        	print 'Unable to process {} due to ValueError, returning negative ones'\
                .format(self.file_name)
        	return np.ones(742)*-1


    def return_frequencies(self):
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

def reduce_one_file(file_name):
    '''
    Reuces one test or train recording by file name
    '''
    print 'Fitting {}'.format(file_name)
    temp_mat = loadmat(file_name)['dataStruct'][0][0]

    patient, id, clas = file_name.replace('.', '_').replace('/','_').split('_')[4:7]
    contaminated = file_name.split('/')[-1] in label
    if patient == 'new': # in case of test data
         patient, id = file_name.replace('.', '_').replace('/','_').split('_')[6:8]
	 clas = None
    else:
	 sequence = temp_mat[4][0][0] # pulls sequence number (only exists in training set)

    start = 0
    result1 = np.array([])
    for segment in xrange(24000, 264000, 24000):
        subset = temp_mat[0][start:segment,:]
        subset = subset[np.all(subset, axis=1)]
        if len(subset) > 0:
            result1 = np.append(result1, subset.mean(axis=0))
        elif len(result1) > 0:
            result1 = np.append(result1, result1[-16:])
        else:
            result1 = np.append(result1, np.zeros(16))
        start = segment+1
    result1 = result1.reshape(1,-1)

    result2 = np.array([])
    for i in range(16):
        cwtavg = signal.cwt(temp_mat[0][:,i], signal.ricker, frequencies)
        result2 = np.concatenate([result2, (cwtavg**2).mean(axis=1)])
    result2 = result2.reshape(1,-1)
    #result = np.append(result, result2)

    result3 = kurtosis(temp_mat[0]).reshape(1,-1) # adds kurtosis for each channel
    result4 = skew(temp_mat[0]).reshape(1,-1) # adds skew for each channel

    temp_mat2 = temp_mat[0][np.all(temp_mat[0], axis=1)]
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

        entropies = []
        for col in range(16):
            kde = gaussian_kde(temp_mat2[:,col])
            r = np.linspace(min(temp_mat2[:,col]), max(temp_mat2[:,col]),\
                len(temp_mat2[:,col])/1E3)
            entropies.append(entropy(kde.evaluate(r)))

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
			            np.array(entropies),
			            np.array(correlations),
                        corr_mean,
                        corr_var]).flat).reshape(1,-1)
    else:
        result_mom = np.zeros(241).reshape(1, -1) # BE CAREFUL WITH THIS

    try:
    	if clas:
		result = np.concatenate([result1, result2, result3, result4, result_mom,
			np.array([patient, id, sequence, contaminated, clas]).reshape(1,-1)], axis=1)
    	else:
        	result = np.concatenate([result1, result2, result3, result4, result_mom,
                        np.array([patient, id]).reshape(1,-1)], axis=1)
    	return result.flatten().reshape(1,-1)
    except ValueError:
	print 'Unable to process {} due to ValueError, returning negative ones'\
        .format(file_name)
	return np.ones(742)*-1


def return_labels():
    '''
    INPUT: None
    OUTPUT: Returns a list of labels of contaminated files
    '''
    label = pd.read_csv('/data/train_and_test_data_labels_safe.csv')
    return list(label[label['safe'] == 0]['image']) # list of contaminated files


def Feature_wrapper(path):
    ''' 
    INPUT: file path 
    OUTPUT: Features result
    '''
    a = Features(path)
    return a.return_results()

def reduce_parallel():
    '''
    to run on aws, change pool and paths
    '''
    params = [('1', '/data/train_1', 'train', 'data/a_reduced17.csv'),
            ('2', '/data/train_2', 'train', 'data/b_reduced17.csv'),
            ('3', '/data/train_3', 'train', 'data/c_reduced17.csv'),
            ('1', '/data/test_1_new', 'test', 'data/a_test_reduced17.csv'),
            ('2', '/data/test_2_new', 'test', 'data/b_test_reduced17.csv'),
            ('3', '/data/test_3_new', 'test', 'data/c_test_reduced17.csv')]
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
        output = pool.map(Features, ifile)
        pd.DataFrame(np.concatenate(output)).to_csv(dfile, index=False)
        print 'Saved file {}'.format(dfile)


if __name__ == '__main__':
    label = return_labels()
    reduce_parallel()
