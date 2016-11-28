
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, auc, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import listdir
from scipy.io import loadmat
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn import svm

class Models(object):
    def __init__(self, patient, data, test_set):
        '''
        Assigns variables for patient, training and test set including the
            basic transformations to begin modeling
        '''
    	self.patient = patient
        self.data = data
    	self.test_set = test_set

    	self.X = None
    	self.y = None

    	self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.models = []
        self.model_scores = []
    	self.predictions = []
    	self.predictions_test_set = []

    	self.transform()


    def transform(self):
    	'''
    	Performs all transformations for train and test sets by:
            1) splitting train/test sets
            2) normalizing
            3) adding dummy variables for patient
            4) removing zeros (if helpful)
    	'''
    	print '-------- Beginning Transformation --------'
        self.remove_zeros() # does this cause a reduction in the score?
    	self.tt_split()
    	self.normalize_and_add_dummies()
    	print '-------- Tranformation Complete --------'


    def remove_zeros(self):
    	'''
    	INPUT: None
    	OUTPUT: None, updates self.data
    	Looks over the first 160 columns of self.data (the channel means) and removes all rows with all zeros
    	'''
    	print 'Dimensions before starting remove_zeros(): {}'.format(self.data.shape)
    	result = []
    	for row in self.data.iterrows():
    		result.append(all(row[1][:160] == 0))
    	result[:] = [not x for x in result]
    	self.data = self.data[result]
    	print 'Dimensions after performing remove_zeros(): {}'.format(self.data.shape)
        print '-------- Removing Zeros Complete --------'


    def tt_split(self):
        '''
    	This function saves the indexes for test and training set, making
            sure that they are separated by group
        '''
        self.y = np.array(self.data.pop(self.data.columns[-1]))
        self.X_train, self.X_test, self.y_train, self.y_test = \
               train_test_split(np.array(self.data),
               self.y,
               test_size=0.3,
               random_state=123)
        print '-------- Train/Test Split Complete --------'


    def normalize_and_add_dummies(self):
    	'''
    	INPUT: None
    	OUTPUT: None
            Normalizes both training and test data by subtracting the mean and
                dividing by the std. Also adds dummies for patient number
    	'''
    	X_train_patient = self.X_train[:,-1]
    	X_test_patient = self.X_test[:,-1]
    	#test_set_patient = self.test_set[:,-1]

    	self.X_train = ((self.X_train - self.X_train.mean(axis=0)) / self.X_train.std(axis=0))[:,:-1]
    	self.X_test = ((self.X_test - self.X_test.mean(axis=0)) / self.X_test.std(axis=0))[:,:-1]
    	#self.test_set = ((self.test_set - self.test_set.mean(axis=0)) / self.test_set.std(axis=0))[:,:-1]

    	self.X_train = np.concatenate([self.X_train, pd.get_dummies(X_train_patient)], axis=1)
    	self.X_test = np.concatenate([self.X_test, pd.get_dummies(X_test_patient)], axis=1)
    	#self.test_set = np.concatenate([self.test_set, pd.get_dummies(test_set_patient)], axis=1)
        print '-------- Normalization and Dummy Adding Complete --------'


    def fit(self):
    	self.logistic_regression()
        self.random_forest()
        self.xgb_static()
        #self.xgb_grid_search()
    	self.svm_rbf()
        self.svm_linear()
    	# self.svm_grid_search() # Found C:1 and gamma:.01 as best choices at .77 score
        print '-------- Fit for Patient {} Complete --------'.format(self.patient)


    def logistic_regression(self):
        model = LogisticRegression(penalty='l1', class_weight='balanced',
            n_jobs=-1)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=5,
            scoring='roc_auc', n_jobs=-1)
        print 'Cross-validated Logistic Regression train score for patient %s: %0.2f (+/- %0.2f)'%(self.patient, scores.mean(), scores.std() * 2)

        model.fit(self.X_train, self.y_train)
    	prediction = model.predict_proba(self.X_test)[:,1]
    	score = self._score(self.y_test, prediction)
        print 'Logistic Regression test score for patient {}: {}'.format(self.patient, score)

    	self.model_scores.append(score)
        self.models.append(model)
    	self.predictions.append(prediction)

    	# self.predictions_test_set.append(model.predict_proba(self.test_set)[:,1])


    def random_forest(self):
        model = RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight='balanced')
        scores = cross_val_score(model, self.X_train, self.y_train, cv=5,
            scoring='roc_auc', n_jobs=-1)
        print 'Cross-validated Random Forest train score for patient %s: %0.2f (+/- %0.2f)'%(self.patient, scores.mean(), scores.std() * 2)

        model.fit(self.X_train, self.y_train)
    	prediction = model.predict_proba(self.X_test)[:,1]
    	score = self._score(self.y_test, prediction)
        print 'Score for Random Forest for patient {}: {}'.format(self.patient, score)

    	self.model_scores.append(score)
       	self.models.append(model)
    	self.predictions.append(prediction)

    	# self.predictions_test_set.append(model.predict_proba(self.test_set)[:,1])

    def xgb_static(self):
        dtrain = xgb.DMatrix(self.X_train, self.y_train)
        dtest =  xgb.DMatrix(self.X_test)
        param = {'max_depth':5,
		'eta':.2, # step shrink size
		'silent':1,
		'objective':'binary:logistic',
		'subsample':.9,
		'booster': 'gbtree',
		'eval_metric': 'auc',
		'scale_pos_weight':12.45} # chose clase weight from sum negative class over sum of positive class
        num_round = 750
        model = xgb.train(param, dtrain, num_round)

        prediction = model.predict(dtest)
        score = self._score(self.y_test, prediction)
        print 'Score for XGB for patient {}: {}'.format(self.patient, score)

        self.models.append(model)
        self.model_scores.append(score)
    	self.predictions.append(prediction)

    	# test_set = xgb.DMatrix(self.test_set)
    	# self.predictions_test_set.append(model.predict(test_set))

    def xgb_grid_search(self):
        cv_params = {'max_depth': [1, 2, 3, 5, 7],
                    'min_child_weight': [1, 3, 5],
    		    'n_estimators': [100, 500, 750, 1000]}
        ind_params = {'seed': 1, # 'learning_rate': 0.1,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'objective': 'binary:logistic'}
        optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
                        cv_params,
                        scoring = 'roc_auc',
                        cv = 3,
                        n_jobs = -1)
        optimized_GBM.fit(self.X_train, self.y_train)
        prediction = optimized_GBM.predict(self.X_test)
        try:
            score = self._score(prediction, self.y_test)
            print 'Best model prediction score: {}'.format(score)
            self.model_scores.append(score)
        except ValueError:
            print 'Only one class present in y_true. ROC AUC score is not defined in that case.'
        # return optimized_GBM
        self.models.append(optimized_GBM.best_estimator_)
        print 'Completed xgb_grid_search'

    def svm_rbf(self):
        y_train = (self.y_train * 2) - 1
    	y_test = (self.y_test * 2) - 1
    	model = svm.SVC(kernel='rbf', C=1, gamma=.01, probability=True, class_weight='balanced')
        scores = cross_val_score(model, self.X_train, self.y_train, cv=5,
            scoring='roc_auc', n_jobs=-1)
        print 'Cross-validated SVM-RBF train score for patient %s: %0.2f (+/- %0.2f)'%(self.patient, scores.mean(), scores.std() * 2)

    	model.fit(self.X_train, y_train)

    	prediction = model.predict_proba(self.X_test)[:,1]
    	score = self._score(y_test, prediction)
    	print "Score for SVM for patient {}: {}".format(self.patient, score)

    	self.models.append(model)
    	self.model_scores.append(score)
    	self.predictions.append(prediction)

    	# self.predictions_test_set.append(model.predict_proba(self.test_set)[:,1])

    def svm_linear(self):
        y_train = (self.y_train * 2) - 1
    	y_test = (self.y_test * 2) - 1
    	model = svm.SVC(kernel='linear', C=1, gamma=.01, probability=True, class_weight='balanced')
        scores = cross_val_score(model, self.X_train, self.y_train, cv=5,
            scoring='roc_auc', n_jobs=-1)
        print 'Cross-validated SVM-Linear train score for patient %s: %0.2f (+/- %0.2f)'%(self.patient, scores.mean(), scores.std() * 2)

    	model.fit(self.X_train, y_train)
    	prediction = model.predict_proba(self.X_test)[:,1]
    	score = self._score(y_test, prediction)
    	print "Score for SVM for patient {}: {}".format(self.patient, score)

    	self.models.append(model)
    	self.model_scores.append(score)
    	self.predictions.append(prediction)

    	# self.predictions_test_set.append(model.predict_proba(self.test_set)[:,1])

    def svm_grid_search(self):
        y_train = (self.y_train * 2) - 1
        y_test = (self.y_test * 2) - 1
    	model = svm.SVC(kernel='rbf', probability=True, class_weight='balanced', verbose=1)

    	params = {'C':[.001,.01, .1, 1, 10, 100, 1000],
    		'gamma':[.01, .1, 1, 10]}
    	model_gs = GridSearchCV(model, params, cv=5, scoring='roc_auc', n_jobs=-1)

    	print '------- Fitting SVM GridSearchCV -------'
    	model_gs.fit(self.X_train, y_train)
    	print 'Best model found has a score of {} with the parameters {}'.format(model_gs.best_score_, model_gs.best_params_)

    def _score(self, y_true, y_pred):
    	'''
    	INPUT: true and predicted values
    	OUTPUT: ROC area under the curve (returns None in case of ValueError)
    	'''
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            print 'ValueError: Returning None. Check to see that y_true is the first argument passed to _score'

    def plot_ROC(self, models):
        plt.figure()
        lw = 2
        for model in models:
            y_score = model.decision_function(self.X_test)
            fpr, tpr, roc_auc = dict(), dict(), dict()
            fpr, tpr, _ = roc_curve(self.y_test, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label=model.solver)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for patient {}'.format(self.patient))
        plt.legend(loc="lower right")
        plt.show()

    def get_feature_importances(df):
        '''
        INPUT: data frame
        OUTPUT: feature importances using random forest
        '''
        x = np.concatenate([self.X_train, self.X_test])
        y = np.concatenate([self.y_train, self.y_test])
        model = RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight='balanced')
        model.fit(x, y)
        return model.feature_importances_


def morlet(n_points, a):
    '''
    INPUT:
        n_points: int - Number of points in vector.  Will be centered around 0.  In
        	scipy.signal.cwt, this will be the nubmer of points that the returned vector
        	will have
        a: scalar - width parameter of the wavelet, defining its size
    OUTPUT: vector: Normalized array of length n_points in shape of a morlet wavelet

    equation = pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))
    '''
    vec = np.linspace(-10, 10, n_points)
    wave = (np.pi**-0.25) * (np.exp(-0.5*vec**2))*(np.exp(-1j*a*vec) - np.exp(-0.5*a**2))

    return wave/np.sum(wave*np.conj(wave)).real


def create_submission(predict_a, file_name):
    '''
    INPUT: predictions(s) and destination file name.  Can take a matrix of
        predictions (for an ensemble) and will return the average
    OUTPUT: None, saves to csv
    '''
    prediction = np.array(predict_a).mean(axis=0)
    files = []
    for directory in ['test_1_new', 'test_2_new', 'test_3_new']:
        file_list = listdir('/data/'+directory)
        file_list = sorted(file_list, key=lambda num: int(num.replace('.',
            '_').split('_')[2]))
        [files.append(file) for file in file_list]
    prediction_df = pd.DataFrame({'File': files, 'Class': prediction}, columns = ['File', 'Class'])
    prediction_df.to_csv(file_name, index=False)
    print 'Saved file {} with shape {}'.format(file_name, prediction_df.shape)


def import_data(separate=False):
    '''
    INPUT: separate - Boolean, whether to return concatenated or separate data frames
    OUTPUT: combined training and test sets
    '''
    a_df = pd.read_csv('data/a_reduced17.csv')
    b_df = pd.read_csv('data/b_reduced17.csv')
    c_df = pd.read_csv('data/c_reduced17.csv')

    a_df = a_df[a_df['819'] == False].drop(['817', '818', '819'], axis=1)
    b_df = b_df[b_df['819'] == False].drop(['817', '818', '819'], axis=1)
    c_df = c_df[c_df['819'] == False].drop(['817', '818', '819'], axis=1)

    a_test = pd.read_csv('data/a_test_reduced16.csv').sort_values(by='818')\
        .drop('818', axis=1)
    b_test = pd.read_csv('data/b_test_reduced16.csv').sort_values(by='818')\
        .drop('818', axis=1)
    c_test = pd.read_csv('data/c_test_reduced16.csv').sort_values(by='818')\
        .drop('818', axis=1)

    if separate:
        return a_df, b_df, c_df, np.array(a_test), np.array(b_test),\
            np.array(c_test)
    else:
        df_concat = pd.concat([a_df, b_df, c_df]).reset_index(drop=True)
        test_concat = np.concatenate([a_test, b_test, c_test])
        return df_concat, test_concat


if __name__ == '__main__':
    df_concat, test_concat = import_data()
    a_df, b_df, c_df, a_test, b_test, c_test = import_data(separate=True)

    com = Models('combined', df_concat, test_concat)
    am = Models('A', a_df, a_test)
    bm = Models('B', b_df, b_test)
    cm = Models('C', c_df, c_test)

    com.fit()
    am.fit()
    bm.fit()
    cm.fit()


    # create_submission(cm.predictions_test_set[0], 'data/prediction20.csv')

    # b.plot_ROC(b.models)
