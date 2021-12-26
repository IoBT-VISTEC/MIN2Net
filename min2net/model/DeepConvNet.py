import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import numpy as np
import os
import time
from sklearn.metrics import classification_report, f1_score
from min2net.utils import TimeHistory, compute_class_weight

class DeepConvNet:
    def __init__(self,
                input_shape=(20,400,1),
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
                model_name='DeepConvNet',
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
        self.Chans = input_shape[0]
        self.Samples = input_shape[1]
        self.kernLength = 125
        self.F1 = 8
        self.D = 2
        self.F2 = int(self.F1*self.D)
        self.norm_rate = 0.25
        self.dropout_rate = 0.5
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
        self.class_weight = None

        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        K.set_image_data_format(self.data_format)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def build(self):
        """ Keras implementation of the Deep Convolutional Network as described in
        Schirrmeister et. al. (2017), Human Brain Mapping.

        This implementation assumes the input is a 2-second EEG signal sampled at
        128Hz, as opposed to signals sampled at 250Hz as described in the original
        paper. We also perform temporal convolutions of length (1, 5) as opposed
        to (1, 10) due to this sampling rate difference.

        Note that we use the max_norm constraint on all convolutional layers, as
        well as the classification layer. We also change the defaults for the
        BatchNormalization layer. We used this based on a personal communication
        with the original authors.

                          ours        original paper
        pool_size        1, 2        1, 3
        strides          1, 2        1, 3
        conv filters     1, 5        1, 10

        Note that this implementation has not been verified by the original
        authors.

        """

        # start the model
        input_main   = Input(self.input_shape)
        block1       = Conv2D(25, (1, 5),
                              input_shape=(self.input_shape),
                              kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
        block1       = Conv2D(25, (self.Chans, 1),
                              kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block1       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
        block1       = Activation('elu')(block1)
        block1       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
        block1       = Dropout(self.dropout_rate)(block1)

        block2       = Conv2D(50, (1, 5),
                              kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block2       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
        block2       = Activation('elu')(block2)
        block2       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
        block2       = Dropout(self.dropout_rate)(block2)

        block3       = Conv2D(100, (1, 5),
                              kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
        block3       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
        block3       = Activation('elu')(block3)
        block3       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
        block3       = Dropout(self.dropout_rate)(block3)

        block4       = Conv2D(200, (1, 5),
                              kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
        block4       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
        block4       = Activation('elu')(block4)
        block4       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
        block4       = Dropout(self.dropout_rate)(block4)

        flatten      = Flatten()(block4)

        dense        = Dense(self.num_class, kernel_constraint = max_norm(0.5))(flatten)
        softmax      = Activation('softmax')(dense)

        return Model(inputs=input_main, outputs=softmax)

    def fit(self, X_train, y_train, X_val, y_val):

        if X_train.ndim != 4:
            raise Exception('ValueError: `X_train` is incompatible: expected ndim=4, found ndim='+str(X_train.ndim))
        elif X_val.ndim != 4:
            raise Exception('ValueError: `X_val` is incompatible: expected ndim=4, found ndim='+str(X_val.ndim))

        self.input_shape = X_train.shape[1:]
        self.Chans = self.input_shape[0]
        self.Samples = self.input_shape[1]
        
        csv_logger = CSVLogger(self.csv_dir)
        time_callback = TimeHistory(self.time_log)
        checkpointer = ModelCheckpoint(monitor=self.monitor, filepath=self.weights_dir, verbose=self.verbose, 
                                       save_best_only=self.save_best_only, save_weight_only=self.save_weight_only)
        reduce_lr = ReduceLROnPlateau(monitor=self.monitor, patience=self.patience, factor=self.factor, mode=self.mode, 
                                      verbose=self.verbose, min_lr=self.min_lr)
        es = EarlyStopping(monitor=self.monitor, mode=self.mode, verbose=self.verbose, patience=self.es_patience)

        model = self.build()
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        model.summary()
        
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
