import min2net.preprocessing as prep
from min2net.preprocessing.config import CONSTANT

k_folds = 5
pick_smp_freq = 100
n_components=4
bands = [[4, 8], [8, 12], [12, 16], 
            [16, 20], [20, 24], [24, 28], 
            [28, 32], [32, 36], [36, 40]]
n_features=8
order = 5
save_path = 'datasets'
num_class = 2

prep.SMR_BCI.fbcsp.subject_dependent_setting(k_folds=k_folds,
                                             pick_smp_freq=pick_smp_freq, 
                                             n_components=n_components,
                                             bands=bands, 
                                             n_features=n_features,
                                             order=order, 
                                             save_path=save_path, 
                                             num_class=num_class, 
                                             sel_chs=CONSTANT['SMR_BCI']['sel_chs'])

prep.BCIC2a.fbcsp.subject_dependent_setting(k_folds=k_folds,
                                            pick_smp_freq=pick_smp_freq, 
                                            n_components=n_components,
                                            bands=bands, 
                                            n_features=n_features,
                                            order=order, 
                                            save_path=save_path, 
                                            num_class=num_class, 
                                            sel_chs=CONSTANT['BCIC2a']['sel_chs'])

prep.OpenBMI.fbcsp.subject_dependent_setting(k_folds=k_folds,
                                             pick_smp_freq=pick_smp_freq, 
                                             n_components=n_components,
                                             bands=bands, 
                                             n_features=n_features,
                                             order=order, 
                                             save_path=save_path, 
                                             num_class=num_class, 
                                             sel_chs=CONSTANT['OpenBMI']['sel_chs'])


prep.SMR_BCI.fbcsp.subject_independent_setting(k_folds=k_folds,
                                               pick_smp_freq=pick_smp_freq, 
                                               n_components=n_components,
                                               bands=bands, 
                                               n_features=n_features,
                                               order=order, 
                                               save_path=save_path, 
                                               num_class=num_class, 
                                               sel_chs=CONSTANT['SMR_BCI']['sel_chs'])

prep.BCIC2a.fbcsp.subject_independent_setting(k_folds=k_folds,
                                              pick_smp_freq=pick_smp_freq, 
                                              n_components=n_components,
                                              bands=bands, 
                                              n_features=n_features,
                                              order=order, 
                                              save_path=save_path, 
                                              num_class=num_class, 
                                              sel_chs=CONSTANT['BCIC2a']['sel_chs'])

prep.OpenBMI.fbcsp.subject_independent_setting(k_folds=k_folds,
                                               pick_smp_freq=pick_smp_freq, 
                                               n_components=n_components,
                                               bands=bands, 
                                               n_features=n_features,
                                               order=order, 
                                               save_path=save_path, 
                                               num_class=num_class, 
                                               sel_chs=CONSTANT['OpenBMI']['sel_chs'])











