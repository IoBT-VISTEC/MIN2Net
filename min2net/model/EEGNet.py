import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, BatchNormalization, Activation, MaxPooling2D, DepthwiseConv2D, AveragePooling2D, SeparableConv2D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import numpy as np
import os
import time
from sklearn.metrics import classification_report, f1_score
from min2net.utils import TimeHistory, compute_class_weight

class EEGNet:
    def __init__(self,
                input_shape=(1,20,400),
                num_class=2,
                loss='sparse_categorical_crossentropy',
                epochs=200,
                batch_size=100,
                optimizer = Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                lr=0.01,
                min_lr=0.01,
                factor=0.25,
                patience=10,
                es_patience=20,
                verbose=1,
                log_path='logs',
                model_name='EEGNet',
                **kwargs):
        self.input_shape = input_shape
        self.num_class = num_class
        self.loss = loss
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
        self.kernLength = 200
        self.F1 = 8
        self.D = 2
        self.F2 = int(self.F1*self.D)
        self.norm_rate = 0.25
        self.dropout_rate = 0.5
        self.f1_average = 'binary' if self.num_class == 2 else 'macro'
        self.data_format = 'channels_first'
        self.shuffle = False
        self.metrics = 'accuracy'
        self.monitor = 'val_loss'
        self.mode = 'min'
        self.save_best_only = True
        self.save_weight_only = True
        self.seed = 1234
        self.class_balancing = False
        self.class_weight = None

        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
            
        if self.data_format == 'channels_first':
            self.Chans = self.input_shape[1]
            self.Samples = self.input_shape[2]
        else:
            self.Chans = self.input_shape[0]
            self.Samples = self.input_shape[1]

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        K.set_image_data_format(self.data_format)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def build(self):
        input1       = Input(shape=self.input_shape)

        ##################################################################
        block1       = Conv2D(self.F1, (1, self.kernLength), padding='same',
                              input_shape=self.input_shape,
                              use_bias=False)(input1)
        block1       = BatchNormalization()(block1)
        block1       = DepthwiseConv2D((self.Chans, 1), use_bias=False,
                                       depth_multiplier=self.D,
                                       depthwise_constraint=max_norm(1.))(block1)
        block1       = BatchNormalization()(block1)
        block1       = Activation('elu')(block1)
        block1       = AveragePooling2D((1, 4))(block1)
        block1       = Dropout(self.dropout_rate)(block1)

        block2       = SeparableConv2D(self.F2, (1, self.kernLength//4),
                                       use_bias=False, padding='same')(block1)
        block2       = BatchNormalization()(block2)
        block2       = Activation('elu')(block2)
        block2       = AveragePooling2D((1, 8))(block2)
        block2       = Dropout(self.dropout_rate)(block2)

        flatten      = Flatten(name='flatten')(block2)

        dense        = Dense(self.num_class, name='dense',
                             kernel_constraint=max_norm(self.norm_rate))(flatten)
        softmax      = Activation('softmax', name='softmax')(dense)

        return Model(inputs=input1, outputs=softmax)

    def fit(self, X_train, y_train, X_val, y_val):

        if X_train.ndim != 4:
            raise Exception('ValueError: `X_train` is incompatible: expected ndim=4, found ndim='+str(X_train.ndim))
        elif X_val.ndim != 4:
            raise Exception('ValueError: `X_val` is incompatible: expected ndim=4, found ndim='+str(X_val.ndim))

        self.input_shape = X_train.shape[1:]
        if self.data_format == 'channels_first':
            self.Chans = self.input_shape[1]
            self.Samples = self.input_shape[2]
        else:
            self.Chans = self.input_shape[0]
            self.Samples = self.input_shape[1]
        
        csv_logger = CSVLogger(self.csv_dir)
        time_callback = TimeHistory(self.time_log)
        checkpointer = ModelCheckpoint(monitor=self.monitor, filepath=self.weights_dir, verbose=self.verbose, 
                                       save_best_only=self.save_best_only, save_weight_only=self.save_weight_only)
        reduce_lr = ReduceLROnPlateau(monitor=self.monitor, patience=self.patience, factor=self.factor, 
                                      mode=self.mode, verbose=self.verbose, min_lr=self.min_lr)
        es = EarlyStopping(monitor=self.monitor, mode=self.mode, verbose=self.verbose, patience=self.es_patience)

        model = self.build()
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        model.summary()
        print("The first kernel size is (1, {})".format(self.kernLength))
        
        if self.class_balancing: # compute_class_weight if class_balancing is True
            self.class_weight = compute_class_weight(y_train)
        else:
            self.class_weight = None         
            
        model.fit(X_train, y_train,
                  batch_size=self.batch_size, shuffle=self.shuffle,
                  epochs=self.epochs, validation_data=(X_val, y_val), class_weight=self.class_weight,
                  callbacks=[checkpointer,csv_logger,reduce_lr,es, time_callback])

    def predict(self, X_test, y_test):

        if X_test.ndim != 4:
            raise Exception('ValueError: `X_test` is incompatible: expected ndim=4, found ndim='+str(X_test.ndim))

        model = self.build()
        model.load_weights(self.weights_dir)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        start = time.time()
        y_pred = model.predict(X_test)
        end = time.time()
        loss, accuracy = model.evaluate(x=X_test, y=y_test, batch_size=self.batch_size, verbose=self.verbose)
        y_pred_argm = np.argmax(y_pred, axis=1)
        print(classification_report(y_test, y_pred_argm))
        print("F1-score is computed based on {}".format(self.f1_average))
        f1 = f1_score(y_test, y_pred_argm, average=self.f1_average)
        evaluation = {'loss': loss, 
                      'accuracy': accuracy, 
                      'f1-score': f1, 
                      'prediction_time': end-start}
        Y = {'y_true': y_test, 
             'y_pred': y_pred_argm}
        return Y, evaluation