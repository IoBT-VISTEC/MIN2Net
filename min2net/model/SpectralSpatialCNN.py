import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Input, Flatten, Dropout, Concatenate
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import numpy as np
import os
import time
from sklearn.metrics import classification_report, f1_score
from min2net.utils import TimeHistory, compute_class_weight

class SpectralSpatialCNN:
    def __init__(self,
                input_shape=(28,28,1),
                num_class=2,
                loss='sparse_categorical_crossentropy',
                epochs=200,
                batch_size=100,
                optimizer = Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                lr=1e-5,
                min_lr=1e-5,
                factor=0.25,
                patience=10,
                es_patience=20,
                verbose=1,
                log_path='logs',
                model_name='SpectralSpatialCNN',
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
        self.n_subbands = 20
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

            
    def __cnn_model(self, model_input): 
        # DeepConvNet
        model = Conv2D(filters=10, kernel_size=(3, 3), activation='relu', padding="same")(model_input) 
        model = Conv2D(filters=14, kernel_size=(3, 3), activation='relu', padding="same")(model)
        model = Conv2D(filters=18, kernel_size=(3, 3), activation='relu', padding="same")(model)
        model = Flatten()(model)
        model = Dense(256)(model)
        return model

    def build(self): 
        input_data    = [Input(shape=self.input_shape) for i in range(self.n_subbands)]
        model_sub     = [self.__cnn_model(input_data[i]) for i in range(self.n_subbands)] 
        out_sub_model = Concatenate()([model_sub[i] for i in range(self.n_subbands)])
        sub_model     = Dense(1024)(out_sub_model)
        sub_model     = Dropout(self.dropout_rate)(sub_model)
        sub_model     = Dense(self.num_class, activation='softmax')(sub_model)
        model         = Model(inputs=input_data, outputs=sub_model)
        return model

    def fit(self, X_train, y_train, X_val, y_val):
        if X_train.ndim != 5:
            raise Exception('ValueError: `X_train` is incompatible: expected ndim=5, found ndim='+str(X_train.ndim))
        elif X_val.ndim != 5:
            raise Exception('ValueError: `X_val` is incompatible: expected ndim=5, found ndim='+str(X_val.ndim))
        
        self.input_shape = (X_train.shape[2], X_train.shape[3], X_train.shape[4])
        self.n_subbands = X_train.shape[1]
        
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
            
        model.fit([X_train[:,i,:,:,:] for i in range(self.n_subbands)], y_train,
                batch_size=self.batch_size,  shuffle=self.shuffle, class_weight=self.class_weight,
                epochs=self.epochs, validation_data=([X_val[:,i,:,:,:] for i in range(self.n_subbands)], y_val),
                callbacks=[checkpointer,csv_logger,reduce_lr,es,time_callback])

    def predict(self, X_test, y_test):
        if X_test.ndim != 5:
            raise Exception('ValueError: `X_test` is incompatible: expected ndim=5, found ndim='+str(X_test.ndim))
        model = self.build()
        model.load_weights(self.weights_dir)
        self.n_subbands = X_test.shape[1]
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        start = time.time()
        y_pred = model.predict([X_test[:,i,:,:,:] for i in range(self.n_subbands)])
        end = time.time()
        loss, acc = model.evaluate(x=[X_test[:,i,:,:,:] for i in range(self.n_subbands)], y=y_test, 
                                   batch_size=self.batch_size, verbose=self.verbose)
        y_pred_argm = np.argmax(y_pred, axis=1)
        print("F1-score is comptured basen on {}".format(self.f1_average))
        f1 = f1_score(y_test, y_pred_argm, average=self.f1_average)
        print(classification_report(y_test, y_pred_argm))
        evaluation = {'loss': loss, 
                      'accuracy': acc, 
                      'f1-score': f1, 
                      'prediction_time': end-start}
        Y = {'y_true': y_test, 
             'y_pred': y_pred_argm}
        return Y, evaluation