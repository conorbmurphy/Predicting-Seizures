from itertools import combinations
import multiprocessing
import numpy as np
from os import listdir
import pandas as pd
from scipy import signal
from scipy.io import loadmat
from scipy.stats import kurtosis, skew, pearsonr, gaussian_kde, entropy


class Features(object):
    '''
    The features class can be use to featurize an iEEG recording.  It
    initializes with a file name to be featurized.

    By calling return_results() it will return the following 817 features for
    the training set (less for the test set as it does not have segment
    recordings or class):

        Channel Means:          0:160       160 channel means by 1 min window
        Wavelet Transforms:     160:560     400 wavelet transformations
        Method of Moments:      560:678     118 statistical moments
        Entropy Calculations:   678:694     16 channel entropies
        Pearson Correleation:   694:816     122 correlations w/ mean and var
        Patient Numbers:        816         1 Patient number
    '''

    def __init__(self, file_name):
        '''
        INPUT: string - file_name
        OUTPUT: None
            Creates variables and runs fit() function
        '''
        self.file_name = file_name
        self.istest = False

        self.temp_mat = None  # Full recording
        self.temp_mat2 = None  # Non-zero values from temp_mat

        self.patient = None
        self.id = None
        self.clas = None
        self.contaminated = None
        self.sequence = None
        self.istest = False  # defaults to False unless changed

        self.means = None
        self.wavelets = None
        self.mom = None
        self.entropies = None
        self.correlations = None

        self.isempty = False

        self.fit()

    def fit(self):
        '''
        INPUT: None
        OUTPUT: None
            Runs load file, saves True if loaded data is empty, and saves
                medatadata
        '''
        print 'Fitting {}'.format(self.file_name)
        self.load_file()

        if self.file_name.split('/')[-1].split('_')[0] == 'new':
            self.istest = True

        self.isempty = self.temp_mat[0].sum() == 0

        self.metadata()

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
        try:
            self.patient, self.id, self.clas = self.file_name.replace('.',
                '_').replace('/', '_').split('_')[4:7]
            self.contaminated = self.file_name.split('/')[-1] in label
            self.sequence = self.temp_mat[4][0][0]
        except IndexError:  # Thrown with test set
            self.patient, self.id = self.file_name.replace('.', '_')\
                .replace('/', '_').split('_')[6:8]
            self.clas = None  # resets class as none
            self.istest = True

    def channel_means(self):
        '''
        INPUT: None
        OUTPUT: 1x160 numpy array
            Divides the sample into 10 segments and returns a 1x160 numpy array
            of the channel means of all non-zero values to self.means()
        '''
        start = 0
        result = np.array([])
        for segment in xrange(24000, 264000, 24000):
            subset = self.temp_mat[0][start:segment, :]
            subset = subset[np.all(subset, axis=1)]
            if len(subset) > 0:
                result = np.append(result, subset.mean(axis=0))
            elif len(result) > 0:
                result = np.append(result, result[-16:])
            else:
                result = np.append(result, np.zeros(16))
            start = segment + 1
        return result.reshape(1, -1)

    def wavelet_transformation(self):
        '''
        INPUT: None
        OUTPUT: 1x400 numpy array
            Performs a wavelet transformation using 25 freqencies of the
            Ricker wave on each channel, saving the means of the squared
            result.
            This returns a 1x400 numpy array, 25 values for each channel
        '''
        freq = self._return_frequencies()
        result = np.array([])
        for i in range(16):  # throws error w/ temp_mat2
            cwtavg = signal.cwt(self.temp_mat[0][:, i], signal.ricker, freq)
            result = np.concatenate([result, (cwtavg**2).mean(axis=1)])
        return result.reshape(1, -1)

    def method_of_moments(self):
        '''
        INPUT: None
        OUTPUT: 1x118 numpy array
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
            variance_of_variances = variance_channel.var(
                ddof=1)  # this is very high

            kurtosis_channel = kurtosis(self.temp_mat2)
            skew_channel = skew(self.temp_mat2)

            max_channel = self.temp_mat2.max(axis=0)
            max_total = self.temp_mat2.max()
            min_channel = self.temp_mat2.min(axis=0)
            min_total = self.temp_mat2.min()

            median_channel = np.median(self.temp_mat2, axis=0)
            median_total = np.median(self.temp_mat2)

            return np.hstack(np.array([
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
                median_total]).flat).reshape(1, -1)

        else:
            return (np.ones(118) * -1).reshape(1, -1)  # BE CAREFUL WITH THIS

    def entropize(self):
        '''
        INPUT: None
        OUTPUT: 1x16 numpy array
            Saves 16 channel entropies to self.entropies, zeros if empty
            dataset
        '''
        entropies = []
        try:
            for col in range(16):
                kde = gaussian_kde(self.temp_mat2[:, col])
                r = np.linspace(np.min(self.temp_mat2[:, col]),
                                np.max(self.temp_mat2[:, col]), 20)
                delt = r[1] - r[0]
                entropies.append(entropy(kde.pdf(r) * delt))
                # ientropies.append((kde.pdf(r)*np.log(kde.pdf(r))).sum()*delt)
            return np.array(entropies).reshape(1, -1)
        except ValueError:  # in case of all zeros
            return (np.ones(16) * -1).reshape(1, -1)

    def correlate(self):
        '''
        INPUT: None
        OUTPUT: 1x122 numpy array
            Saves 120 combinations of channel pearson correlations, their mean
            and variance
        '''
        correlations = []
        for c in combinations(range(16), 2):
            correlations.append(pearsonr(self.temp_mat2[:, c[0]],
                                         self.temp_mat2[:, c[1]])[0])
        corr_mean = np.mean(correlations)
        corr_var = np.var(correlations)
        return np.hstack([np.array(correlations), corr_mean, corr_var])\
            .reshape(1, -1)

    def return_train(self):
        '''
        INPUT: None
        OUTPUT: 1x821 numpy array of compiled train data
        '''
        result = np.concatenate([
            self.channel_means(),
            self.wavelet_transformation(),
            self.method_of_moments(),
            self.entropize(),
            self.correlate(),
            np.array([
                self.patient,
                self.id,
                self.sequence,
                self.contaminated,
                self.clas]).reshape(1, -1)], axis=1)
        return result.flatten().reshape(1, -1)

    def return_test(self):
        '''
        INPUT: None
        OUTPUT: 1x818 numpy array of compiled test data
        '''
        result = np.concatenate([
            self.channel_means(),
            self.wavelet_transformation(),
            self.method_of_moments(),
            self.entropize(),
            self.correlate(),
            np.array([
                self.patient,
                self.id]).reshape(1, -1)], axis=1)
        return result.flatten().reshape(1, -1)

    def return_results(self):
        '''
        INPUT: None
        OUTPUT: Returns either test or training compelation
        '''
        if not self.istest:
            return self.return_train()
        elif (not self.isempty) and (self.istest):
            return self.return_test()
        elif self.istest:
            return (np.ones(818) * -1).reshape(1, -1)
        else:
            return (np.ones(821) * -1).reshape(1, -1)

    def _return_frequencies(self):
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
        frequencies = np.concatenate([
            np.linspace(300, 100, 5),  # delta waves
            np.linspace(95, 57, 5),  # theta waves
            np.linspace(55, 27, 5),  # alpha waves
            np.linspace(25, 13, 5),  # beta waves
            np.linspace(12, 2, 5)])  # gamma waves
        return frequencies


def return_labels():
    '''
    INPUT: None
    OUTPUT: Returns a list of labels of contaminated files
    '''
    label = pd.read_csv('/data/train_and_test_data_labels_safe.csv')
    # list of contaminated files
    return list(label[label['safe'] == 0]['image'])


def Feature_wrapper(path):
    '''
    INPUT: file path
    OUTPUT: Features result
        In case of error, return all -1's and append path to global list
        'errors'
    '''
    try:
        a = Features(path)
        result = a.return_results()
        print 'Returning {} with shape {}'.format(path, result.shape)
        if (result.shape == (1, 821)) or (result.shape == (1, 818)):
            return result
        else:
            return (np.ones(821) * -1).reshape(-1, 1)
    except:
        print 'Unexpected error on {}'.format(path)
        return (np.ones(821) * -1).reshape(-1, 1)
        errors.append(path)


def reduce_parallel(params, n_jobs=1):
    '''
    INPUT: n_jobs - int (optional) - number of cores to use
           params - list of tuples of length 4 - list of tuples with the order
           (patient_number(str), directory(str), test_or_train(str),
           dest_file(str))
    OUTPUT: None
        Parallelizes feature building across the cores specified in n_jobs
            and saves result in the directory 'data' (not to be confused with
            the root directory /data)
        To change the location of iEEG files or name of destination files,
            revise params
    '''
    files = [listdir(param[1]) for param in params]
    files[0].remove('1_45_1.mat')  # removes corrupt file
    paths = [param[1] for param in params]
    dest_file = [param[3] for param in params]
    outputs = []
    for i in range(6):
        dfile = dest_file[i]
        ifile = [paths[i] + '/' + file for file in files[i]]
        print 'Launching pool to construct {}'.format(dfile)
        pool = multiprocessing.Pool(n_jobs)
        output = pool.map(Feature_wrapper, ifile)
        pd.DataFrame(np.concatenate(output)).to_csv(dfile, index=False)
        print 'Saved file {}'.format(dfile)


if __name__ == '__main__':
    errors = []
    label = return_labels()
    params = [('1', '/data/train_1', 'train', 'data/a_reduced.csv'),
              ('2', '/data/train_2', 'train', 'data/b_reduced.csv'),
              ('3', '/data/train_3', 'train', 'data/c_reduced.csv'),
              ('1', '/data/test_1_new', 'test', 'data/a_test_reduced.csv'),
              ('2', '/data/test_2_new', 'test', 'data/b_test_reduced.csv'),
              ('3', '/data/test_3_new', 'test', 'data/c_test_reduced.csv')]

    reduce_parallel(params, n_jobs=40)

    print '{} Errors in fitting files'.format(len(errors))
    for i, error in enumerate(errors):
        print 'Error 1 with file {}'.format(i, error)
