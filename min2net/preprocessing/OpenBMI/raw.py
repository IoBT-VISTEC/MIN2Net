import numpy as np
import scipy.io as sio
from min2net.utils import resampling
from min2net.preprocessing.config import CONSTANT
from sklearn.model_selection import train_test_split
CONSTANT = CONSTANT['OpenBMI']

def read_raw(PATH, session, subject, num_class, id_ch_selected):
    mat_file_name = PATH + '/sess'+str(session).zfill(2)+'_subj'+str(subject).zfill(2)+'_EEG_MI.mat'
    mat = sio.loadmat(mat_file_name)
    print('This is data from: ', mat_file_name)
    if num_class == 2:
        raw_train_data = mat['EEG_MI_train'][0]['smt'][0]
        raw_train_data = (np.swapaxes(raw_train_data, 0, 2))[id_ch_selected]
        raw_train_data = np.swapaxes(raw_train_data, 0, 1)
        print('raw_train_data_shape:', raw_train_data.shape)
        raw_test_data = mat['EEG_MI_test'][0]['smt'][0]
        raw_test_data = np.swapaxes(raw_test_data, 0, 2)[id_ch_selected]
        raw_test_data = np.swapaxes(raw_test_data, 0, 1)
        print('raw_test_data_shape:', raw_test_data.shape)
        label_train_data = mat['EEG_MI_train'][0]['y_dec'][0][0]-1
        label_test_data = mat['EEG_MI_test'][0]['y_dec'][0][0]-1
        return raw_train_data, label_train_data, raw_test_data, label_test_data

    elif num_class == 3:
        raw_train_data = __segment_data(mat, type_data = 'train')
        raw_train_data = np.take(raw_train_data, id_ch_selected, axis=2)
        raw_train_data = np.swapaxes(raw_train_data, 1, 2)
        print('raw_train_data_shape:', raw_train_data.shape)
        raw_test_data = __segment_data(mat, type_data = 'test')
        raw_test_data = np.take(raw_test_data, id_ch_selected, axis=2)
        raw_test_data = np.swapaxes(raw_test_data, 1, 2)
        print('raw_test_data_shape:', raw_test_data.shape)
        label_train_data = mat['EEG_MI_train'][0]['y_dec'][0][0]-1
        label_test_data = mat['EEG_MI_test'][0]['y_dec'][0][0]-1
        return raw_train_data, label_train_data, raw_test_data, label_test_data
    
    elif num_class == "transitory_mi":
        raw_train_data = __segment_data_whole_period(mat, type_data = 'train')
        raw_train_data = np.take(raw_train_data, id_ch_selected, axis=2)
        raw_train_data = np.swapaxes(raw_train_data, 1, 2)
        print('raw_train_data_shape:', raw_train_data.shape)
        raw_test_data = __segment_data_whole_period(mat, type_data = 'test')
        raw_test_data = np.take(raw_test_data, id_ch_selected, axis=2)
        raw_test_data = np.swapaxes(raw_test_data, 1, 2)
        print('raw_test_data_shape:', raw_test_data.shape)
        label_train_data = mat['EEG_MI_train'][0]['y_dec'][0][0]-1
        label_test_data = mat['EEG_MI_test'][0]['y_dec'][0][0]-1
        return raw_train_data, label_train_data, raw_test_data, label_test_data

def load_crop_data(PATH, n_subjs, new_smp_freq, num_class, MI_len, id_chosen_chs, start_mi=None, stop_mi=None):
    if num_class == 2:
        print("Two-class MI data is downloading")
        orig_smp_freq = CONSTANT['orig_smp_freq'] # 1000
        n_trials = CONSTANT['n_trials_2_class'] # 100
        sessions = [1, 2]
        n_chs = len(id_chosen_chs)
        X_train, y_train = np.zeros((n_subjs,len(sessions),n_trials,n_chs,int(new_smp_freq* MI_len))), np.zeros((n_subjs,len(sessions),n_trials))
        X_test, y_test = np.zeros((n_subjs,len(sessions),n_trials,n_chs,int(new_smp_freq* MI_len))), np.zeros((n_subjs,len(sessions),n_trials))
        for id_sub, subject in enumerate(range(1, n_subjs+1)):
            for id_se, sess in enumerate(sessions):
                X_tr, y_tr, X_te, y_te = read_raw(PATH, sess, subject, num_class, id_chosen_chs)
                X_tr_resam = resampling(X_tr, new_smp_freq, MI_len)
                X_te_resam = resampling(X_te, new_smp_freq, MI_len)
                X_train[id_sub, id_se, :, :, :] = X_tr_resam
                X_test[id_sub, id_se, :, :, :] = X_te_resam
                y_train[id_sub, id_se, :] = y_tr
                y_test[id_sub, id_se, :] = y_te
        return X_train.reshape(n_subjs,-1,n_chs,int(new_smp_freq* MI_len)), y_train.reshape(n_subjs,-1), X_test.reshape(n_subjs,-1,n_chs,int(new_smp_freq* MI_len)), y_test.reshape(n_subjs,-1)
    elif num_class == 3:
        print("Three-class MI data is downloading")
        orig_smp_freq = CONSTANT['orig_smp_freq'] #1000
        n_trials = CONSTANT['n_trials_3_class'] #150
        sessions = [1, 2]
        n_chs = len(id_chosen_chs)
        MI_len = 4
        X_train, y_train = np.zeros((n_subjs,len(sessions),n_trials,n_chs,int(new_smp_freq* MI_len))), np.zeros((n_subjs,len(sessions),n_trials))
        X_test, y_test = np.zeros((n_subjs,len(sessions),n_trials,n_chs,int(new_smp_freq* MI_len))), np.zeros((n_subjs,len(sessions),n_trials))
        for id_sub, subject in enumerate(range(1, n_subjs+1)):
            for id_se, sess in enumerate(sessions):
                X_tr, y_tr, X_te, y_te = read_raw(PATH, sess, subject, num_class, id_chosen_chs)
                X_tr_addon, y_tr_addon = __add_on_resting(X_tr, y_tr, orig_smp_freq)
                X_te_addon, y_te_addon = __add_on_resting(X_te, y_te, orig_smp_freq)
                X_tr_resam = resampling(X_tr_addon, new_smp_freq, MI_len)
                X_te_resam = resampling(X_te_addon, new_smp_freq, MI_len)
                X_train[id_sub, id_se, :, :, :] = X_tr_resam
                X_test[id_sub, id_se, :, :, :] = X_te_resam
                y_train[id_sub, id_se, :] = y_tr_addon
                y_test[id_sub, id_se, :] = y_te_addon
        return X_train.reshape(n_subjs,-1,n_chs,int(new_smp_freq* MI_len)), y_train.reshape(n_subjs,-1), X_test.reshape(n_subjs,-1,n_chs,int(new_smp_freq* MI_len)), y_test.reshape(n_subjs,-1)
    
    elif num_class == "transitory_mi" and start_mi != None and stop_mi != None:
        print("Two-class transitory MI data is downloading with the time interval of {} s and {} s".format(start_mi, stop_mi))
        orig_smp_freq = CONSTANT['orig_smp_freq'] # 1000
        n_trials = CONSTANT['n_trials_2_class'] # 100
        sessions = [1, 2]
        n_chs = len(id_chosen_chs)
        MI_len = 4
        X_test, y_test = np.zeros((n_subjs,len(sessions),n_trials,n_chs,int(new_smp_freq* MI_len))), np.zeros((n_subjs,len(sessions),n_trials))
        for id_sub, subject in enumerate(range(1, n_subjs+1)):
            for id_se, sess in enumerate(sessions):
                X_tr, y_tr, X_te, y_te = read_raw(PATH, sess, subject, num_class, id_chosen_chs)
                X_te_transient, y_te_transient = __transitory_mi(X_te, y_te, orig_smp_freq, start_mi, stop_mi)
                X_te_resam = resampling(X_te_transient, new_smp_freq, MI_len)
                X_test[id_sub, id_se, :, :, :] = X_te_resam
                y_test[id_sub, id_se, :] = y_te_transient
        return X_test.reshape(n_subjs,-1,n_chs,int(new_smp_freq* MI_len)), y_test.reshape(n_subjs,-1)
                
def chanel_selection(sel_chs):
    orig_chs = CONSTANT['orig_chs']
    chs_id = []
    for name_ch in sel_chs:
        ch_id = np.where(np.array(orig_chs) == name_ch)[0][0]
        chs_id.append(ch_id)
        print('chosen_channel:', name_ch ,'---', 'Index_is:', ch_id)
    return chs_id

def __segment_data(mat_arr, type_data):
    data = mat_arr['EEG_MI_'+type_data][0]['x'][0]
    t = mat_arr['EEG_MI_'+type_data][0]['t'][0][0]
    low_cut = 0
    high_cut = CONSTANT['trial_len'] #8s
    orig_smp_freq = CONSTANT['orig_smp_freq'] #1000
    orig_n_chs = CONSTANT['n_chs'] #62
    n_trials = CONSTANT['n_trials_2_class'] #100
    data_seg = np.zeros((n_trials, high_cut*orig_smp_freq, orig_n_chs))
    # print('This pre-processing is for task {} low cut {} high cut {}'.format(task,low_cut,high_cut))
    for i in range(n_trials):
        start_pos = t[i]+(low_cut*orig_smp_freq)
        stop_pos = t[i]+(high_cut*orig_smp_freq)
        data_seg[i, :, :] = data[start_pos:stop_pos, :]
    return data_seg

def __segment_data_whole_period(mat_arr, type_data):
    data = mat_arr['EEG_MI_'+type_data][0]['x'][0]
    t = mat_arr['EEG_MI_'+type_data][0]['t'][0][0]
    low_cut = -3
    high_cut = CONSTANT['trial_len'] #8s
    data_len = 11
    orig_smp_freq = CONSTANT['orig_smp_freq'] #1000
    orig_n_chs = CONSTANT['n_chs'] #62
    n_trials = CONSTANT['n_trials_2_class'] #100
    data_seg = np.zeros((n_trials, data_len*orig_smp_freq, orig_n_chs))
    # print('This pre-processing is for task {} low cut {} high cut {}'.format(task,low_cut,high_cut))
    for i in range(n_trials):
        start_pos = t[i]+(low_cut*orig_smp_freq)
        stop_pos = t[i]+(high_cut*orig_smp_freq)
        # print("Debugg the selected period is:", (stop_pos-start_pos)/orig_smp_freq)
        data_seg[i, :, :] = data[start_pos:stop_pos, :]
    return data_seg

def __add_on_resting(X, y, smp_freq):
    print("MI Right, MI Left and Resting EEG Segmentation Process is being processed...")
    print("This data contains {} time ponts with sampling frequency of {} Hz.".format(X.shape[2], smp_freq))
    start_pos_mi = int(CONSTANT['MI']['start']*smp_freq) #0s
    stop_pos_mi = int(CONSTANT['MI']['stop']*smp_freq) #4s
    start_pos_rest = int(CONSTANT['MI']['stop']*smp_freq) #4s
    stop_pos_rest = int(CONSTANT['trial_len']*smp_freq) #8s
    index_class1 = np.where(y == 0)[0]
    index_class2 = np.where(y == 1)[0]
    X_class1, y_class1 = X[index_class1], y[index_class1]
    X_class2, y_class2 = X[index_class2], y[index_class2]
    # Split data into resting and MI signals
    X_mi_class1 = X_class1[:,:,start_pos_mi:stop_pos_mi]
    X_mi_class2 = X_class2[:,:,start_pos_mi:stop_pos_mi]
    X_rest_class1 = X_class1[:,:,start_pos_rest:stop_pos_rest]
    X_rest_class2 = X_class2[:,:,start_pos_rest:stop_pos_rest]
    # Choose a half of resting data to keep balancing the number of classes in our data
    X_rest_class1_50per,_ ,_ ,_= train_test_split(X_rest_class1, y_class1, random_state=42, test_size=0.5)
    X_rest_class2_50per,_ ,_ ,_= train_test_split(X_rest_class2, y_class2, random_state=42, test_size=0.5)
    X_rest_all = np.concatenate((X_rest_class1_50per, X_rest_class2_50per), axis=0)
    # Build class for resting data
    y_rest_all = np.full(X_rest_all.shape[0], 2)
    # Combine all classes again
    X_new_all = np.concatenate((X_mi_class1, X_mi_class2, X_rest_all), axis=0)
    y_new_all = np.concatenate((y_class1, y_class2, y_rest_all), axis=0)
    return X_new_all, y_new_all

def __transitory_mi(X, y, smp_freq, start, stop):
    print("MI Right ang MI Left EEG including transitory period is being processed...")
    print("This data contains {} time ponts with sampling frequency of {} Hz.".format(X.shape[2], smp_freq))
    start_pos_mi = int(start*smp_freq)
    stop_pos_mi = int(stop*smp_freq)
    # Segment needed MI period
    X_mi = X[:,:,start_pos_mi:stop_pos_mi]
    return X_mi, y
