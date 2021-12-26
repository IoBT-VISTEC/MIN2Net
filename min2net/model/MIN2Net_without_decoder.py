from tensorflow.keras.layers import Concatenate, AveragePooling2D, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Input, Reshape, Flatten
from tensorflow.keras import models
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np
import time
import os
from sklearn.metrics import classification_report, f1_score
from min2net.loss import triplet_loss, SparseCategoricalCrossentropy
from min2net.utils import TimeHistory, compute_class_weight

class MIN2Net_without_decoder:
    def __init__(self,
                input_shape=(1,400,20), 
                num_class=2, 
                loss=[triplet_loss(margin=1.0), 'sparse_categorical_crossentropy'],
                loss_weights=[1., 1.], 
                latent_dim = None,
                epochs=200,
                batch_size=100,
                optimizer = Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                lr=1e-2,
                min_lr=1e-3,
                factor=0.25,
                patience=20, 
                es_patience=50,
                verbose=1,
                log_path='logs',
                model_name='MIN2Net_without_decoder', 
                **kwargs):
        D, T, C = input_shape
        self.latent_dim = latent_dim if latent_dim is not None else C if num_class==2 else 64
        self.num_class = num_class
        self.input_shape = input_shape
        self.loss = loss
        self.loss_weights = loss_weights
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.optimizer.lr = lr
        self.lr = lr
        self.min_lr = min_lr
        self.factor = factor
        self.patience = patience
        self.es_patience = es_patience
        self.verbose = verbose
        self.log_path = log_path
        self.model_name = model_name
        self.weights_dir = log_path+'/'+model_name+'_out_weights.h5'
        self.csv_dir = log_path+'/'+model_name+'_out_log.log'
        self.time_log = log_path+'/'+model_name+'_time_log.csv'

        # use **kwargs to set the new value of below args.
        self.f1_average = 'binary' if self.num_class == 2 else 'macro'
        self.data_format = 'channels_last'
        self.shuffle = False
        self.metrics = 'accuracy'
        self.monitor = 'val_loss'
        self.mode = 'min'
        self.save_best_only = True
        self.save_weight_only = True
        self.seed = 1234
        self.class_balancing = False
        # 'set params'
        self.subsampling_size = 100
        self.subsampling_size = 100
        self.pool_size_1 = (1,T//self.subsampling_size)
        self.pool_size_2 = (1,4)
        self.filter_1 = C
        self.filter_2 = 10
        
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
        
        self.flatten_size = T//self.pool_size_1[1]//self.pool_size_2[1]
        
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        K.set_image_data_format(self.data_format)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def build(self):
        'encoder --> DCNN'
        encoder_input  = Input(self.input_shape)
        en_conv        = Conv2D(self.filter_1, (1, 64), activation='elu', padding="same", 
                                kernel_constraint=max_norm(2., axis=(0, 1, 2)))(encoder_input)
        en_conv        = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(en_conv)
        en_conv        = AveragePooling2D(pool_size=self.pool_size_1)(en_conv)  
        en_conv        = Conv2D(self.filter_2, (1, 32), activation='elu', padding="same", 
                                kernel_constraint=max_norm(2., axis=(0, 1, 2)))(en_conv)
        en_conv        = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(en_conv)
        en_conv        = AveragePooling2D(pool_size=self.pool_size_2)(en_conv)
        en_conv        = Flatten()(en_conv)
        encoder_output = Dense(self.latent_dim, kernel_constraint=max_norm(0.5))(en_conv)
   

        z              = Dense(self.num_class, activation='softmax', kernel_constraint=max_norm(0.5), 
                               name='classifier')(encoder_output)

        return models.Model(inputs=encoder_input, outputs=[encoder_output, z], 
                            name='MIN2Net_without_decoder')

    
    def fit(self, X_train, y_train, X_val, y_val):

        if X_train.ndim != 4:
            raise Exception('ValueError: `X_train` is incompatible: expected ndim=4, found ndim={}'.format(X_train.ndim))
        elif X_val.ndim != 4:
            raise Exception('ValueError: `X_val` is incompatible: expected ndim=4, found ndim={}'.format(X_val.ndim))

        csv_logger = CSVLogger(self.csv_dir)
        time_callback = TimeHistory(self.time_log)
        checkpointer = ModelCheckpoint(monitor=self.monitor, filepath=self.weights_dir, 
                                       verbose=self.verbose, save_best_only=self.save_best_only, 
                                       save_weight_only=self.save_weight_only)
        reduce_lr = ReduceLROnPlateau(monitor=self.monitor, patience=self.patience, 
                                      factor=self.factor, mode=self.mode, verbose=self.verbose, 
                                      min_lr=self.min_lr)
        es = EarlyStopping(monitor=self.monitor, mode=self.mode, verbose=self.verbose, 
                           patience=self.es_patience)
        model = self.build()
        model.summary()
        
        if self.class_balancing: # compute_class_weight if class_balancing is True
            class_weight  = compute_class_weight(y_train)
            self.loss[-1] = SparseCategoricalCrossentropy(class_weight=class_weight)
            
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics, loss_weights=self.loss_weights)

        model.fit(x=X_train, y=[y_train, y_train],
                    batch_size=self.batch_size, shuffle=self.shuffle, 
                    epochs=self.epochs, validation_data=(X_val, [y_val,y_val]),
                    callbacks=[checkpointer,csv_logger,reduce_lr,es, time_callback])
        
    def predict(self, X_test, y_test):  

        if X_test.ndim != 4:
            raise Exception('ValueError: `X_test` is incompatible: expected ndim=4, found ndim={}'.format(X_test.ndim))

        model = self.build()
        model.load_weights(self.weights_dir)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics, loss_weights=self.loss_weights)

        start = time.time()
        y_pred_trip, y_pred_clf = model.predict(X_test)
        end = time.time()
        loss, trip_loss, classifier_loss, trip_acc, classifier_acc  = model.evaluate(x=X_test,
                                                                                     y=[y_test,y_test],
                                                                                     batch_size=self.batch_size, 
                                                                                     verbose=self.verbose)
        
        y_pred_argm = np.argmax(y_pred_clf, axis=1)
        f1 = f1_score(y_test, y_pred_argm, average=self.f1_average)
        print("F1-score is computed based on {}".format(self.f1_average))
        print('(loss: {}, accuracy: {})'.format(loss, classifier_acc))
        print(classification_report(y_test, y_pred_argm))
        evaluation = {'loss': loss, 
                      'triplet_loss': trip_loss, 
                      'classifier_loss': classifier_loss, 
                    'accuracy': classifier_acc, 
                    'f1-score': f1, 
                    'prediction_time': end-start}
        Y = {'y_true': y_test, 
             'y_pred': y_pred_argm}

        return Y, evaluation