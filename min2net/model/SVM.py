from sklearn.metrics import classification_report, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from joblib import dump, load
import numpy as np
import os
import time
from min2net.utils import write_log

class SVM:
    def __init__(self,
                log_path='logs',
                model_name='SVM',
                tuned_parameters=None,
                num_class=2,
                **kwargs):
        if tuned_parameters == None:
            self.tuned_parameters = [{'kernel': ['rbf'],
                                      'gamma': [1e-2, 1e-3],
                                      'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                                     },
                                    {'kernel': ['sigmoid'],
                                     'gamma': [1e-2, 1e-3],
                                     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                                    },
                                    {'kernel': ['linear'],
                                     'gamma': [1e-2, 1e-3],
                                     'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]
                                    }]
        else:
            self.tuned_parameters = tuned_parameters
        self.seed = 1234
        self.num_class = num_class
        self.f1_average = 'binary' if self.num_class == 2 else 'macro'

        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
            
        np.random.seed(self.seed)

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.model_path = log_path+'/'+model_name+'_model.gz'
        self.time_log = log_path+'/'+model_name+'_time_log.csv'

    def fit(self, X_train, y_train, X_val, y_val):

        if X_train.ndim != 2:
            raise Exception('ValueError: `X_train` is incompatible: expected ndim=4, found ndim='+str(X_train.ndim))
        elif X_val.ndim != 2:
            raise Exception('ValueError: `X_val` is incompatible: expected ndim=4, found ndim='+str(X_val.ndim))

        print('Dimension of training set is: {} and label is: {}'.format(X_train.shape, y_train.shape))
        print('Dimension of validation set is: {} and label is: {}'.format(X_val.shape, y_val.shape))

        X_all = np.concatenate((X_train, X_val),axis=0)
        y_all = np.concatenate((y_train, y_val),axis=0)

        # Create a list where train data indices are -1 and validation data indices are 0
        tr_index = np.full((X_train.shape[0]), -1)
        val_index = np.full((X_val.shape[0]), 0)
        split_index = np.concatenate((tr_index, val_index), axis=0).tolist()
        # Use the list to create PredefinedSplit
        pds = PredefinedSplit(test_fold = split_index)
        clf = GridSearchCV(estimator=SVC(), param_grid=self.tuned_parameters, cv=pds, scoring = 'accuracy')
        start = time.time()
        clf.fit(X_all , y_all)
        end = time.time()
        #Clasifying with an optimal parameter set
        Optimal_params = clf.best_params_
        print(Optimal_params)
        classifier = SVC(**Optimal_params)
        classifier.fit(X_train, y_train)
        dump(classifier, self.model_path)
        write_log(filepath=self.time_log, data=['time_log'], mode='w')
        write_log(filepath=self.time_log, data=[end-start], mode='a')

    def predict(self, X_test, y_test):

        if X_test.ndim != 2:
            raise Exception('ValueError: `X_test` is incompatible: expected ndim=4, found ndim='+str(X_test.ndim))

        print('Dimesion of testing set is: {} and label is: {}'.format(X_test.shape, y_test.shape))
        classifier = load(self.model_path)
        svm_acc = classifier.score(X_test, y_test)
        print('===Test Score for Optimized Parameters===:', svm_acc)
        start = time.time()
        y_true, y_pred = y_test, classifier.predict(X_test)
        end = time.time()
        print(classification_report(y_true, y_pred))
        n_classes = len(np.unique(y_test))
        print("F1-score is computed based on {}".format(self.f1_average))
        f1 = f1_score(y_true, y_pred, average=self.f1_average)
        evaluation = {'accuracy': svm_acc, 
                      'f1-score': f1, 
                      'prediction_time': end-start}
        Y = {'y_true': y_true, 
             'y_pred': y_pred}
        return Y, evaluation
