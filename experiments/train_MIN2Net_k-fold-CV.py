import tensorflow as tf
import numpy as np
import os
import argparse
from tensorflow.keras.optimizers import Adam
from min2net.model import MIN2Net
from min2net.utils import write_log, DataLoader
from min2net.loss import mean_squared_error, triplet_loss
from config import config

'''
----------------------------
python train_MIN2Net_k-fold-CV.py \
    --dataset 'OpenBMI' \
    --train_type 'subject_independent' --GPU 1 \
    --margin 1.0 --loss_weights 0.5 0.5 1.0

'''

def k_fold_cross_validation(subject):
    # create object of DataLoader
    loader = DataLoader(dataset=args.dataset, 
                        train_type=args.train_type, 
                        subject=subject, 
                        data_format=data_format, 
                        data_type=data_type, 
                        dataset_path=args.data_path)

    y_true, y_pred, y_pred_decoder = [], [], []
    for fold in range(1, n_folds+1):

        model_name = 'S{:03d}_fold{:02d}'.format(subject, fold)
        model = MIN2Net(input_shape=input_shape,
                        class_balancing=True,
                        f1_average='binary',
                        num_class=num_class,
                        loss=[mean_squared_error, triplet_loss(margin=args.margin), 'sparse_categorical_crossentropy'],
                        loss_weights=args.loss_weights,
                        epochs=epochs,
                        batch_size=batch_size,
                        optimizer=Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                        lr=lr,
                        min_lr=min_lr,
                        factor=factor,
                        patience=patience,
                        es_patience=es_patience,
                        latent_dim = latent_dim,
                        log_path=log_path,
                        model_name=model_name)

        # load dataset
        X_train, y_train = loader.load_train_set(fold=fold)
        X_val, y_val = loader.load_val_set(fold=fold)
        X_test, y_test = loader.load_test_set(fold=fold)

        # train and test using MIN2Net
        model.fit(X_train, y_train, X_val, y_val)
        Y, evaluation = model.predict(X_test, y_test)

        # logging
        csv_file = log_path+'/S{:03d}_all_results.csv'.format(subject)
        if fold==1:
            write_log(csv_file, data=evaluation.keys(), mode='w')
        write_log(csv_file, data=evaluation.values(), mode='a')
        y_true.append(Y['y_true'])
        y_pred.append(Y['y_pred'])
        y_pred_decoder.append(Y['y_pred_decoder'])
        tf.keras.backend.clear_session()

    # writing results
    np.savez(log_path+'/S{:03d}_Y_results.npz'.format(subject),
            y_true=np.array(y_true),
            y_pred=np.array(y_pred))
    np.save(log_path+'/S{:03d}_Y_reconstruction.npy'.format(subject),
            np.array(y_pred_decoder))
    print('------------------------- S{:03d} Done--------------------------'.format(subject))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_weights', nargs='+', default=[0.5, 0.5, 1.0], type=float, help='loss_weights (beta): ex. [beta1,beta2,beta3]')
    parser.add_argument('--save_path', type=str, default='logs/MIN2Net', help='path to save logs')
    parser.add_argument('--data_path', type=str, default='datasets', help='path to datasets')
    parser.add_argument('--dataset', type=str, default='OpenBMI', help='dataset name: ex. [BCIC2a/SMR_BCI/OpenBMI]')
    parser.add_argument('--subject', nargs='+', default=None, type=int, help='list of test subject id, None=all subject')
    parser.add_argument('--train_type', type=str, default=None, help='Train type: ex. subject_dependent, subject_independent')
    parser.add_argument('--GPU', type=str, default='0', help='GPU ID')
    parser.add_argument('--margin', type=float, default=1.0, help='margin (alpha)')
    parser.add_argument('--num_class', type=int, default=2, help='number of classes')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
    print(tf.config.experimental.list_physical_devices('GPU'))

    # load config params from config.py
    CONSTANT = config['MIN2Net']
    factor = CONSTANT[args.train_type]['factor']
    es_patience = CONSTANT[args.train_type]['es_patience']
    lr = CONSTANT[args.train_type]['lr']
    min_lr = CONSTANT[args.train_type]['min_lr']
    batch_size = CONSTANT[args.train_type]['batch_size']
    patience = CONSTANT[args.train_type]['patience']
    epochs = CONSTANT[args.train_type]['epochs']
    n_folds = CONSTANT[args.train_type]['n_folds']
    data_type = CONSTANT[args.train_type]['data_type']
    n_subjects = CONSTANT[args.dataset]['n_subjects']
    input_shape = CONSTANT[args.dataset]['input_shape']
    data_format = CONSTANT[args.dataset]['data_format']
    num_class = CONSTANT[args.dataset]['num_class']
    num_class = args.num_class if args.num_class != 2 else num_class
    latent_dim = input_shape[2] if num_class == 2 else 64 # n_channels or 64

    print('TRAIN SET: {}'.format(args.dataset))
    print('The size of latent vector: {}'.format(latent_dim))
    
    log_path = '{}_margin{}/{}_{}_classes_{}_{}'.format(args.save_path, str(args.margin), args.train_type, str(num_class), args.dataset, str(args.loss_weights))
    for directory in [log_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    if args.subject == None: #loop to train all subjects
        for subject in range(1, n_subjects+1):
            k_fold_cross_validation(subject)
    else:
        #train a sigle subject
        for subject in args.subject:
            k_fold_cross_validation(subject)