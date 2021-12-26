from min2net.utils import PATH

CONSTANT = {
    'BCIC2a': {
        'raw_path': 'datasets/BCIC2a/raw', # raw data path 'raw_path': 'datasets/BCIC2a'
        'n_subjs': 9,
        'n_trials': 144,
        'n_trials_per_class': 72,
        'n_chs': 20,
        'orig_smp_freq': 250,                  # Original sampling frequency (Hz)
        'trial_len': 7,                        # 7s
        'MI': {
            'start': 2,                        # start at time = 2 s
            'stop': 6,                         # stop at time = 6 s
            'len': 4,                          # 4s
        },
        'orig_chs': ['FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                    'P1', 'Pz', 'P2'],
        'sel_chs': ['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 
                    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                    'P1', 'Pz', 'P2'] 
    },
    'SMR_BCI': {
        'raw_path': 'datasets/SMR_BCI/raw', # raw data path
        'n_subjs': 14,
        'n_trials_tr': 100,
        'n_trials_te': 60, 
        'n_chs': 15,
        'orig_smp_freq': 512,                   # Original sampling frequency  (Hz)
        'trial_len': 8,                         # 7s
        'MI': {
            'start': 4,                         # start at time = 4 s
            'stop': 8,                          # stop at time = 8 s
            'len': 4,                           # 4s
        },
        'orig_chs': ['FCC3',                   'FCCz',                 'FCC4',
                    'C5h', 'C3', 'C3h',       'C1h', 'Cz', 'C2h',       'C4h', 'C4', 'C6h',
                           'CCP3',                   'CCPz',                 'CCP4'],
        'sel_chs': [       'FCC3',                   'FCCz',                 'FCC4', 
                    'C5h', 'C3', 'C3h',       'C1h', 'Cz', 'C2h',       'C4h','C4', 'C6h', 
                           'CCP3',                   'CCPz',                 'CCP4']  
    },
    'OpenBMI': {
        'raw_path': 'datasets/OpenBMI/raw', # raw data path
        'n_subjs': 54,
        'n_trials_2_class': 100,
        'n_trials_3_class': 150, 
        'n_chs': 62,
        'orig_smp_freq': 1000,                  # Original sampling frequency  (Hz)
        'trial_len': 8,                         # 8s (cut-off)
        'MI': {
            'start': 0,                         # start at time = 0 s
            'stop': 4,                          # stop at time = 0 s
            'len': 4,                           # 4s
        },
        'orig_chs': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 
                    'C3','Cz','C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 
                    'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1',
                    'C2', 'C6', 'CP3','CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 
                    'TP7', 'TPP9h', 'FT10','FTT10h','TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 
                    'AF7', 'AF3', 'AF4', 'AF8', 'PO3','PO4'],
        'sel_chs': ['FC5', 'FC3', 'FC1', 'FC2', 'FC4','FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 
                    'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']  
    }

}