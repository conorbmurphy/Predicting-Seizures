import numpy as np
from scipy import signal
from scipy.io import loadmat
import multiprocessing
from os import listdir
from sklearn.decomposition import PCA

# cwtmatr = cwt(mat_channel, signal.ricker, np.linspace(103, 193, 20, endpoint=False))

# cwtmatr = cwt(mat_channel, signal.ricker, np.linspace(103, 193, 20, endpoint=False))
# plt.imshow(cwtmatr, extent=[0, 240000, 103, 193], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())

def wavelet_spectrogram(file):
    mat = loadmat(file)['dataStruct'][0][0][0]
    name = file.replace('/', '.').split('.')[-2]
    print 'Starting {}'.format(file)
    result = np.array([])
    for i in range(16):
        cwtmatr = signal.cwt(mat[:,i], signal.ricker, np.linspace(50, 500, 20, endpoint=False))
        pca = PCA(n_components=3)
    	if result.shape[0] == 0:
        	result = pca.fit_transform(cwtmatr)
    	else:
        	result = np.concatenate([result, pca.fit_transform(cwtmatr)])
    np.save('/data/conv/'+name, result)

def wavelet_spectrogram2(file):
	'''
	This is creates an average across channels
	'''
    mat = loadmat(file)['dataStruct'][0][0][0]
    name = file.replace('/', '.').split('.')[-2]
    print 'Starting {}'.format(file)
    result2 = np.array([])
    for i in range(16):
        cwtavg = signal.cwt(mat[:,i], signal.ricker, np.linspace(50, 500, 20, endpoint=False)).mean(axis=1)
	result2 = np.concatenate([result2, cwtavg])
    np.save('/data/conv/'+name, result2)

def rand_10_train(location):
    files = listdir(location)
    return [location+'/'+file for file in files]

def rand_20_test(location):
    files = listdir(location)
    return [location+'/'+file for file in files]


def random_files():
    locations_train = ['/data/train_1', '/data/train_2', '/data/train_3']
    location_test = ['/data/test_1_new', '/data/test_2_new', '/data/test_3_new']
    files = []
    for location in locations_train:
        files.append(rand_10_train(location))
    for location in location_test:
        files.append(rand_20_test(location))
    return [item for sublist in files for item in sublist]

def plot_pool():
    files = random_files()
    pool = multiprocessing.Pool(40)
    output = pool.map(wavelet_spectrogram, files)

# 3 hz 'spike and wave' - multiply by cos(2pi times t(5sec) / cycle length (1/3 for 3 hz).  Then multiply by a wavelengeth that is the length of the space.  Multiply that by every 5 sec window (or whatever)

if __name__ == '__main__':
    plot_pool()

# zip -r images.zip images

