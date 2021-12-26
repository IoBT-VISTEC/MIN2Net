config = {
    'MIN2Net': {
            'subject_dependent': {
                    'factor': 0.5,
                    'es_patience': 20,
                    'lr': 0.001,
                    'min_lr': 0.0001,
                    'batch_size': 10,
                    'patience': 5,
                    'epochs': 200,
                    'n_folds': 5,
                    'data_type':'time_domain'
            },
            'subject_independent': {
                    'factor': 0.5,
                    'es_patience': 20,
                    'lr': 0.001,
                    'min_lr': 0.0001,
                    'batch_size': 100,
                    'patience': 5,
                    'epochs': 200,
                    'n_folds': 5,
                    'data_type':'time_domain'
            },
            'BCIC2a': {
                    'n_subjects': 9,
                    'input_shape': (1,400,20),
                    'data_format': 'NTCD',
                    'num_class': 2,
            },
            'OpenBMI': {
                    'n_subjects': 54,
                    'input_shape': (1,400,20),
                    'data_format': 'NTCD',
                    'num_class': 2
             },
            'SMR_BCI': {
                    'n_subjects': 14,
                    'input_shape': (1,400,15),
                    'data_format': 'NTCD',
                    'num_class': 2
            }
    },

    'MIN2Net_without_triplet': {
            'subject_dependent': {
                    'factor': 0.5,
                    'es_patience': 20,
                    'lr': 0.001,
                    'min_lr': 0.0001,
                    'batch_size': 10,
                    'patience': 5,
                    'epochs': 200,
                    'n_folds': 5,
                    'data_type':'time_domain'        
            },
            'subject_independent': {
                    'factor': 0.5,
                    'es_patience': 20,
                    'lr': 0.001,
                    'min_lr': 0.0001,
                    'batch_size': 100,
                    'patience': 5,
                    'epochs': 200,
                    'n_folds': 5,
                    'data_type':'time_domain'
            },
            'BCIC2a': {
                    'n_subjects': 9,
                    'input_shape': (1,400,20),
                    'data_format': 'NTCD',
                    'num_class': 2
            },
            'OpenBMI': {
                    'n_subjects': 54,
                    'input_shape': (1,400,20),
                    'data_format': 'NTCD',
                    'num_class': 2
            },
            'SMR_BCI': {
                    'n_subjects': 14,
                    'input_shape': (1,400,15),
                    'data_format': 'NTCD',
                    'num_class': 2
            }
    },
    'MIN2Net_without_decoder': {
            'subject_dependent': {
                    'factor': 0.5,
                    'es_patience': 20,
                    'lr': 0.001,
                    'min_lr': 0.0001,
                    'batch_size': 10,
                    'patience': 5,
                    'epochs': 200,
                    'n_folds': 5,
                    'data_type':'time_domain'
            },
            'subject_independent': {
                    'factor': 0.5,
                    'es_patience': 20,
                    'lr': 0.001,
                    'min_lr': 0.0001,
                    'batch_size': 100,
                    'patience': 5,
                    'epochs': 200,
                    'n_folds': 5,
                    'data_type':'time_domain'
            },
            'BCIC2a': {
                    'n_subjects': 9,
                    'input_shape': (1,400,20),
                    'data_format': 'NTCD',
                    'num_class': 2
            },
            'OpenBMI': {
                    'n_subjects': 54,
                    'input_shape': (1,400,20),
                    'data_format': 'NTCD',
                    'num_class': 2
            },
            'SMR_BCI': {
                    'n_subjects': 14,
                    'input_shape': (1,400,15),
                    'data_format': 'NTCD',
                    'num_class': 2
            }
    },

    'DeepConvNet': {
            'subject_dependent': {
                    'factor': 0.5,
                    'es_patience': 20,
                    'lr': 0.01,
                    'min_lr': 0.01,
                    'batch_size': 10,
                    'patience': 5,
                    'epochs': 200,
                    'n_folds': 5,
                    'dropout_rate': 0.5,
                    'data_type':'time_domain'
            },
            'subject_independent': {
                    'factor': 0.5,
                    'es_patience':20,
                    'lr': 0.01,
                    'min_lr': 0.01,
                    'batch_size': 100,
                    'patience': 5,
                    'epochs': 200,
                    'n_folds': 5,
                    'dropout_rate': 0.5,
                    'data_type':'time_domain'
            },
            'BCIC2a': {
                    'n_subjects': 9,
                    'input_shape': (20,400,1),
                    'data_format': 'NCTD',
                    'num_class': 2
            },
            'OpenBMI': {
                    'n_subjects': 54,
                    'input_shape': (20,400,1),
                    'data_format': 'NCTD',
                    'num_class': 2
            },
            'SMR_BCI': {
                    'n_subjects': 14,
                    'input_shape': (15,400,1),
                    'data_format': 'NCTD',
                    'num_class': 2
            },
    },

    'EEGNet': {
            'subject_dependent': {
                    'factor': 0.5,
                    'es_patience': 20,
                    'lr': 0.01,
                    'min_lr': 0.01,
                    'batch_size': 10,
                    'patience': 5,
                    'epochs': 200,
                    'n_folds': 5,
                    'dropout_rate': 0.5,
                    'data_type':'time_domain'
            },
            'subject_independent': {
                    'factor': 0.5,
                    'es_patience': 20,
                    'lr': 0.01,
                    'min_lr': 0.01,
                    'batch_size': 100,
                    'patience': 5,
                    'epochs': 200,
                    'n_folds': 5,
                    'dropout_rate': 0.25,
                    'data_type':'time_domain'
            },
            'BCIC2a': {
                    'n_subjects': 9,
                    'input_shape': (20,400,1),
                    'data_format': 'NCTD',
                    'num_class': 2
            },
            'OpenBMI': {
                    'n_subjects': 54,
                    'input_shape': (20,400,1),
                    'data_format': 'NCTD',
                    'num_class': 2
            },
            'SMR_BCI': {
                    'n_subjects': 14,
                    'input_shape': (15,400,1),
                    'data_format': 'NCTD',
                    'num_class': 2
            }
    },

    'SpectralSpatialCNN': {
            'subject_dependent': {
                    'factor': 0.5,
                    'es_patience': 20,
                    'lr': 0.00001,
                    'min_lr': 0.00001,
                    'batch_size': 10,
                    'patience': 10,
                    'epochs': 200,
                    'n_folds': 5,
                    'dropout_rate': 0.5,
                    'data_type':'spectral_spatial'
            },
            'subject_independent': {
                    'factor': 0.5,
                    'es_patience': 20,
                    'lr': 0.00001,
                    'min_lr': 0.00001,
                    'batch_size': 100,
                    'patience': 5,
                    'epochs': 200,
                    'n_folds': 5,
                    'dropout_rate': 0.5,
                    'data_type':'spectral_spatial'
            },
            'BCIC2a': {
                    'n_subjects': 9,
                    'input_shape': (28,28,1),
                    'data_format': 'NSHWD',
                    'num_class': 2
            },
            'OpenBMI': {
                    'n_subjects': 54,
                    'input_shape': (28,28,1),
                    'data_format': 'NSHWD',
                    'num_class': 2
            },
            'SMR_BCI': {
                    'n_subjects': 14,
                    'input_shape': (23,23,1),
                    'data_format': 'NSHWD',
                    'num_class': 2
            }
    },

        'FBCSP-SVM': {
            'BCIC2a': {
                    'n_subjects': 9,
                    'data_format': None,
                    'num_class': 2
            },
            'OpenBMI': {
                    'n_subjects': 54,
                    'data_format': None,
                    'num_class': 2
            },
            'SMR_BCI': {
                    'n_subjects': 14,
                    'data_format': None,
                    'num_class': 2
            },
            'n_folds': 5,
            'data_type':'fbcsp',
            'tuned_parameters' : [{
                        'kernel': ['rbf'],
                        'gamma': [1e-2, 1e-3],
                        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                        },
                        {
                        'kernel': ['sigmoid'],
                        'gamma': [1e-2, 1e-3],
                        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                        },
                        {
                        'kernel': ['linear'],
                        'gamma': [1e-2, 1e-3],
                        'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]
                        }]

    }

}
