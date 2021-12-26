import numpy as np
import os
from min2net.model import SVM
from min2net.utils import write_log, DataLoader
import argparse
from config import config

'''
python train_FBCSP-SVM_k-fold-CV.py \
    --dataset 'BCIC2a' \
    --train_type 'subject_dependent'

'''

def k_fold_cross_validation(subject):
    # create object of DataLoader
    loader = DataLoader(dataset=args.dataset, 
                        train_type=args.train_type, 
                        subject=subject, 
                        data_format=data_format, 
                        data_type=data_type, 
                        dataset_path=args.data_path)

    y_true, y_pred = [], []
    for fold in range(1, n_folds+1):
        
        model_name='SVM_S{:03d}_fold{:02d}'.format(subject, fold)
        svm = SVM(log_path=log_path, 
                  model_name=model_name,
                  num_class=num_class,
                  tuned_parameters=tuned_parameters)
                  
        # load dataset
        X_train, y_train = loader.load_train_set(fold=fold)
        X_val, y_val = loader.load_val_set(fold=fold)
        X_test, y_test = loader.load_test_set(fold=fold)
       
        # train and test using SVM
        svm.fit(X_train, y_train, X_val, y_val)
        Y, evaluation = svm.predict(X_test, y_test)

        # logging
        csv_file = log_path+'/S{:03d}_all_results.csv'.format(subject)
        if fold==1:
            write_log(csv_file, data=evaluation.keys(), mode='w')
        write_log(csv_file, data=evaluation.values(), mode='a')
        y_true.append(Y['y_true'])
        y_pred.append(Y['y_pred'])

    # writing file
    np.savez(log_path+'/S{:03d}_Y_results.npz'.format(subject),
            y_true=np.array(y_true), 
            y_pred=np.array(y_pred))
    print('------------------------- S{:03d} Done--------------------------'.format(subject))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='logs/FBCSP-SVM', help='path to save logs')
    parser.add_argument('--data_path', type=str, default='datasets', help='path to datasets')
    parser.add_argument('--dataset', type=str, default='OpenBMI', help='dataset name: ex. [BCIC2a/SMR_BCI/OpenBMI]')
    parser.add_argument('--subject', nargs='+', default=None, type=int, help='list of test subject id, None=all subject')
    parser.add_argument('--train_type', type=str, default=None, help='Train type: ex. subject_dependent, subject_independent')
    parser.add_argument('--num_class', type=int, default=2, help='number of classes')
    args = parser.parse_args()

    # load config params from config.py
    CONSTANT = config['FBCSP-SVM']
    n_folds = CONSTANT['n_folds']
    data_type = CONSTANT['data_type']
    tuned_parameters = CONSTANT['tuned_parameters']
    n_subjects = CONSTANT[args.dataset]['n_subjects']
    data_format = CONSTANT[args.dataset]['data_format']
    num_class = CONSTANT[args.dataset]['num_class']

    print('TRAIN SET: {}'.format(args.dataset))

    log_path = '{}/{}_{}_classes'.format(args.save_path, args.train_type, str(num_class), args.dataset)
    for directory in [log_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    if args.subject == None: #loop to train all subjects
        for subject in range(1, n_subjects+1):
            k_fold_cross_validation(subject)
    else:
        for subject in args.subject:
            k_fold_cross_validation(subject)
