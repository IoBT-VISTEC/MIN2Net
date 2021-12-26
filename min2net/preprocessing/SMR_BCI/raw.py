import numpy as np
import scipy.io as sio
from min2net.utils import resampling
from min2net.preprocessing.config import CONSTANT
# load variable form config file
CONSTANT = CONSTANT['SMR_BCI']
orig_chs = CONSTANT['orig_chs']
orig_smp_freq = CONSTANT['orig_smp_freq']
trial_len = CONSTANT['trial_len'] 
n_chs = CONSTANT['n_chs'] # 15
window_len = CONSTANT['trial_len']*CONSTANT['orig_smp_freq'] # 8*512

def read_raw(PATH, subject , training, id_chosen_chs):
    if training:
        mat = sio.loadmat(PATH+'/S'+str(subject).zfill(2)+'T.mat')['data']
        n_trials = CONSTANT['n_trials_tr'] # 100
        label = np.zeros(n_trials)
        data = np.zeros((n_trials, n_chs, window_len))
    else:
        mat = sio.loadmat(PATH+'/S'+str(subject).zfill(2)+'E.mat')['data']
        n_trials = CONSTANT['n_trials_te'] # 60
        label = np.zeros(n_trials)
        data = np.zeros((n_trials, n_chs, window_len))
    NO_valid_trial = 0
    for ii in range(0,mat.size):
        mat_1 = mat[0,ii]
        mat_2 = [mat_1[0,0]]
        mat_info = mat_2[0]
        _X = mat_info[0]
        _trial = mat_info[1]
        _y = mat_info[2]
        _fs = mat_info[3]
        _classes = mat_info[4]
        for trial in range(0, _trial.size):
            # class 1 (right hand) and class 2 (feet) 
            _data = np.transpose(_X[_trial[0][trial]:int(_trial[0][trial]+window_len),id_chosen_chs])
            _label = int(_y[0][trial])
            data[NO_valid_trial,:,:] =  _data
            label[NO_valid_trial] = _label
            NO_valid_trial +=1
    return data, label-1

def chanel_selection(sel_chs): 
    chs_id = []
    for name_ch in sel_chs:
        ch_id = np.where(np.array(orig_chs) == name_ch)[0][0]
        chs_id.append(ch_id)
        print('chosen_channel:', name_ch, '---', 'Index_is:', ch_id)
    return chs_id

def load_crop_data(PATH, subject, start, stop, new_smp_freq, id_chosen_chs):
    start_time = int(start*new_smp_freq) # 4*
    stop_time = int(stop*new_smp_freq) # 8*
    X_train, y_tr = read_raw(PATH=PATH, subject=subject, training=True, id_chosen_chs=id_chosen_chs)
    X_test, y_te = read_raw(PATH=PATH, subject=subject, training=False, id_chosen_chs=id_chosen_chs)
    if new_smp_freq < orig_smp_freq:
        X_train = resampling(X_train, new_smp_freq, trial_len)
        X_test = resampling(X_test, new_smp_freq, trial_len)
    X_train = X_train[:,:,start_time:stop_time]
    X_test = X_test[:,:,start_time:stop_time]
    return X_train, y_tr, X_test, y_te  