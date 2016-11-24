
from sklearn.model_selection import train_test_split, cross_val_score
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
    	self.patient = patient
        self.data = data
    	self.test_set = test_set

    	self.X = None
    	self.y = None

    	#self.train_indexes = [] # Remove?
    	#self.test_indexes = []
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
    	# self.add_groups() # No longer needed - part of feature building
    	self.tt_split()
    	self.normalize_and_add_dummies()
            # self.remove_zeros() # does this cause a reduction in the score?
    	print '-------- Tranformation Complete --------'

    def fit(self):
	self.logistic_regression()
	#self.logistic_regression()
        #self.random_forest()
        #self.xgb_static()
        #self.xgb_grid_search()
	#self.svm_static()
	# self.svm_grid_search() # Found C:1 and gamma:.01 as best choices at .77 score

    def tt_split(self):
        '''
    	This function saves the indexes for test and training set, making
            sure that they are separated by group
        '''
        self.y = self.data.pop(self.data.columns[-1])
        self.X_train, self.X_test, self.y_train, self.y_test = \
               train_test_split(self.data, self.y, test_size=0.4)
        print '-------- Train/Test Split Complete --------'
	# group_kfold = GroupKFold(n_splits=3)
	# self.y = np.array(self.data.pop(self.data.columns[-1]))
	# self.X = np.array(self.data)
    #
	# for train_index, test_index in group_kfold.split(self.X, self.y, self.groups):
	# 	self.train_indexes.append(train_index)
	# 	self.test_indexes.append(test_index)
	# 	#X_train, X_test = self.data[train_index], self.data[test_index]
	# 	#y_train, y_test = self.data[train_index], self.data[test_index]

        # for i in range(1, 7):
        #    subset = self.data[self.data['739'] == i]
        #    if i == 1:
        #        y = subset.pop(subset.columns[-1])
        #        self.X_train, self.X_test, self.y_train, self.y_test = \
        #            train_test_split(subset, y, test_size=0.4)
        #    else:
        #        y = subset.pop(subset.columns[-1])
        #        temp_X_train, temp_X_test, temp_y_train, temp_y_test = \
        #            train_test_split(subset, y, test_size=0.4)
        #        self.X_train = np.concatenate([self.X_train, temp_X_train])
        #        self.X_test = np.concatenate([self.X_test, temp_X_test])
        #        self.y_train = np.concatenate([self.y_train, temp_y_train])
        #        self.y_test = np.concatenate([self.y_test, temp_y_test])
    	# self.X_train = self.X_train[:,:-1] # dropping recording number
    	# self.X_test = self.X_test[:,:-1]

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

    def normalize_and_add_dummies(self):
	'''
	INPUT: None
	OUTPUT: None, normalizes both training and test data by subtracting the mean and dividing by the std
		Also adds dummies for patient number
	'''
	X_patient = self.X[:,-1]
	test_set_patient = self.test_set[:,-1]

	self.X = ((self.X - self.X.mean(axis=0)) / self.X.std(axis=0))[:,:-1]
	self.test_set = ((self.test_set - self.test_set.mean(axis=0)) / self.test_set.std(axis=0))[:,:-1]

	self.X = np.concatenate([self.X, pd.get_dummies(X_patient)], axis=1)
	self.test_set = np.concatenate([self.test_set, pd.get_dummies(test_set_patient)], axis=1)

	#X_train_patient = self.X_train[:,-1]
	#X_test_patient = self.X_test[:,-1]
	#test_set_patient = self.test_set[:,-1]

	#self.X_train = ((self.X_train - self.X_train.mean(axis=0)) / self.X_train.std(axis=0))[:,:-1]
	#self.X_test = ((self.X_test - self.X_test.mean(axis=0)) / self.X_test.std(axis=0))[:,:-1]
	#self.test_set = ((self.test_set - self.test_set.mean(axis=0)) / self.test_set.std(axis=0))[:,:-1]

	#self.X_train = np.concatenate([self.X_train, pd.get_dummies(X_train_patient)], axis=1)
	#self.X_test = np.concatenate([self.X_test, pd.get_dummies(X_test_patient)], axis=1)
	#self.test_set = np.concatenate([self.test_set, pd.get_dummies(test_set_patient)], axis=1)
    print '-------- Normalization and Dummy Adding Complete --------'

    def logistic_regression(self):
	for train_index, test_index in zip(self.train_indexes, self.test_indexes):
        	model = LogisticRegression(penalty='l1', class_weight='balanced', n_jobs=-1)
        	model.fit(self.X[train_index], self.y[train_index])

		prediction = model.predict_proba(self.X[test_index])[:,1]
	        score = self._score(self.y[test_index], prediction)
        	print 'Score for Logistic Regression for patient {}: {}'.format(self.patient, score)

	        self.model_scores.append(score)
        	self.models.append(model)
		self.predictions.append(prediction)

		self.predictions_test_set.append(model.predict_proba(self.test_set)[:,1])

    def random_forest(self):
        model = RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight='balanced')
        model.fit(self.X_train, self.y_train)

	prediction = model.predict_proba(self.X_test)[:,1]
	score = self._score(self.y_test, prediction)
        print 'Score for Random Forest for patient {}: {}'.format(self.patient, score)

	self.model_scores.append(score)
   	self.models.append(model)
	self.predictions.append(prediction)

	self.predictions_test_set.append(model.predict_proba(self.test_set)[:,1])

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

	test_set = xgb.DMatrix(self.test_set)
	self.predictions_test_set.append(model.predict(test_set))

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

    def svm_static(self):
        y_train = (self.y_train * 2) - 1
	y_test = (self.y_test * 2) - 1
	model = svm.SVC(kernel='rbf', C=1, gamma=.01, probability=True, class_weight='balanced')
	model.fit(self.X_train, y_train)

	prediction = model.predict_proba(self.X_test)[:,1]
	score = self._score(y_test, prediction)
	print "Score for SVM for patient {}: {}".format(self.patient, score)

	self.models.append(model)
	self.model_scores.append(score)
	self.predictions.append(prediction)

	self.predictions_test_set.append(model.predict_proba(self.test_set)[:,1])

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

    # def plot_ROC(self, models):
    #     plt.figure()
    #     lw = 2
    #     for model in models:
    #         y_score = model.decision_function(self.X_test)
    #         fpr, tpr, roc_auc = dict(), dict(), dict()
    #         fpr, tpr, _ = roc_curve(self.y_test, y_score)
    #         roc_auc = auc(fpr, tpr)
    #         plt.plot(fpr, tpr, color='darkorange', lw=lw, label=model.solver)
    #     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver operating characteristic for patient {}'.format(self.patient))
    #     plt.legend(loc="lower right")
    #     plt.show()

    def create_final_prediction(self, model):
        '''
        INPUT: model to predict on and test set.  This will refit the model on
            the entire training set before predicting
        OUTPUT: predicted model
        '''
        x = np.concatenate([self.X_train, self.X_test])
        y = np.concatenate([self.y_train, self.y_test])
        try:
            model.fit(x, y)
            print 'Used fit method to create model'
            return model.predict(test_set)
        except AttributeError:
            print 'fitting xgboost'
            dtrain = xgb.DMatrix(x, y)
            dtest = xgb.DMatrix(self.test_set)
	    param = {'max_depth':5,
                'eta':.2, # step shrink size
                'silent':1,
                'objective':'binary:logistic',
                'subsample':.9,
                'booster': 'gbtree',
                'eval_metric': 'auc'}
            num_round = 750
            bst = xgb.train(param, dtrain, num_round)

            return bst.predict(dtest)



def combine_predictions_by_patient(predict_a, predict_b, predict_c, file_name):
    '''
    INPUT: predictions from three patients
    OUTPUT: None, saves to csv
    '''
    prediction = np.concatenate([predict_a, predict_b, predict_c])
    files = []
    for directory in ['test_1_new', 'test_2_new', 'test_3_new']:
        file_list = listdir('/data/'+directory)
        file_list = sorted(file_list, key=lambda num: int(num.replace('.',
            '_').split('_')[2]))
        [files.append(file) for file in file_list]
    prediction_df = pd.DataFrame({'File': files, 'Class': prediction}, columns = ['File', 'Class'])
    prediction_df.to_csv(file_name, index=False)
    print 'Saved file {} with shape {}'.format(file_name, prediction_df.shape)

def combine_predictions_with_combined_patients(predict_a, file_name):
    '''
    INPUT: predictions from combined patients
    OUTPUT: None, saves to csv
    '''
    prediction = predict_a
    files = []
    for directory in ['test_1_new', 'test_2_new', 'test_3_new']:
        file_list = listdir('/data/'+directory)
        file_list = sorted(file_list, key=lambda num: int(num.replace('.',
            '_').split('_')[2]))
        [files.append(file) for file in file_list]
    prediction_df = pd.DataFrame({'File': files, 'Class': prediction}, columns = ['File', 'Class'])
    prediction_df.to_csv(file_name, index=False)
    print 'Saved file {} with shape {}'.format(file_name, prediction_df.shape)

def combine_predictions_ensemble(predict_a, file_name):
    '''
    INPUT: an ensemble method for predictions from combined patients
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

def import_data():
    '''
    INPUT: None
    OUTPUT: combined training and test sets
    '''
    a_df = pd.read_csv('data/a_reduced15.csv')
    b_df = pd.read_csv('data/b_reduced15.csv')
    c_df = pd.read_csv('data/c_reduced15.csv')
    df_concat = pd.concat([\
		a_df[a_df['740'] == False].drop(['738', '739', '740'], axis=1),
		b_df[b_df['740'] == False].drop(['738', '739', '740'], axis=1),
		c_df[c_df['740'] == False].drop(['738', '739', '740'], axis=1)])\
			.reset_index(drop=True)

    a_test = pd.read_csv('data/a_test_reduced15.csv').sort_values(by='738')\
        .drop('738', axis=1)
    b_test = pd.read_csv('data/b_test_reduced15.csv').sort_values(by='738')\
        .drop('738', axis=1)
    c_test = pd.read_csv('data/c_test_reduced15.csv').sort_values(by='738')\
        .drop('738', axis=1)
    test_concat = np.concatenate([a_test, b_test, c_test])

    return df_concat, test_concat

if __name__ == '__main__':
    # a_df = pd.read_csv('data/a_reduced15.csv')
    # b_df = pd.read_csv('data/b_reduced15.csv')
    # c_df = pd.read_csv('data/c_reduced15.csv')
    # df_concat = pd.concat([\
	# 	a_df[a_df['740'] == False].drop(['738', '739', '740'], axis=1),
	# 	b_df[b_df['740'] == False].drop(['738', '739', '740'], axis=1),
	# 	c_df[c_df['740'] == False].drop(['738', '739', '740'], axis=1)])\
	# 		.reset_index(drop=True)
    #
    # a_test = pd.read_csv('data/a_test_reduced15.csv').sort_values(by='738').drop('738', axis=1)
    # b_test = pd.read_csv('data/b_test_reduced15.csv').sort_values(by='738').drop('738', axis=1)
    # c_test = pd.read_csv('data/c_test_reduced15.csv').sort_values(by='738').drop('738', axis=1)
    # test_concat = np.concatenate([a_test, b_test, c_test])

    df_concat, test_concat = import_data()

    # cm = Models('combined', df_concat, test_concat)
    # cm.fit()
    # combine_predictions_ensemble(cm.predictions_test_set[0], 'data/prediction20.csv')


    #combined_combine_predictions(combined_model.create_final_prediction(combined_model.models[0]), 'data/prediction16.csv')

    #combine_predictions(model_a.create_final_prediction(model_a.models[0], a_test),
    #                    model_b.create_final_prediction(model_b.models[0], b_test),
   #                    model_c.create_final_prediction(model_c.models[0], c_test),
    #                    'data/prediction11.csv')
    #pred = pd.read_csv('data/prediction11.csv')

    # b.plot_ROC(b.models)
##
