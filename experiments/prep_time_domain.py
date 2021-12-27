import min2net.preprocessing as prep
from min2net.preprocessing.config import CONSTANT

k_folds = 5
pick_smp_freq = 100
bands = [8, 30]
order = 5
save_path = 'datasets'
num_class = 2

prep.SMR_BCI.time_domain.subject_dependent_setting(k_folds=k_folds,
                                                   pick_smp_freq=pick_smp_freq, 
                                                   bands=bands, 
                                                   order=order, 
                                                   save_path=save_path, 
                                                   num_class=num_class, 
                                                   sel_chs=CONSTANT['SMR_BCI']['sel_chs'])

prep.BCIC2a.time_domain.subject_dependent_setting(k_folds=k_folds,
                                                  pick_smp_freq=pick_smp_freq, 
                                                  bands=bands, 
                                                  order=order, 
                                                  save_path=save_path, 
                                                  num_class=num_class, 
                                                  sel_chs=CONSTANT['BCIC2a']['sel_chs'])

prep.OpenBMI.time_domain.subject_dependent_setting(k_folds=k_folds,
                                                   pick_smp_freq=pick_smp_freq, 
                                                   bands=bands, 
                                                   order=order, 
                                                   save_path=save_path, 
                                                   num_class=num_class, 
                                                   sel_chs=CONSTANT['OpenBMI']['sel_chs'])


prep.SMR_BCI.time_domain.subject_independent_setting(k_folds=k_folds,
                                                     pick_smp_freq=pick_smp_freq, 
                                                     bands=bands, 
                                                     order=order, 
                                                     save_path=save_path, 
                                                     num_class=num_class, 
                                                     sel_chs=CONSTANT['SMR_BCI']['sel_chs'])

prep.BCIC2a.time_domain.subject_independent_setting(k_folds=k_folds,
                                                    pick_smp_freq=pick_smp_freq, 
                                                    bands=bands, 
                                                    order=order, 
                                                    save_path=save_path, 
                                                    num_class=num_class, 
                                                    sel_chs=CONSTANT['BCIC2a']['sel_chs'])

prep.OpenBMI.time_domain.subject_independent_setting(k_folds=k_folds,
                                                     pick_smp_freq=pick_smp_freq, 
                                                     bands=bands, 
                                                     order=order, 
                                                     save_path=save_path, 
                                                     num_class=num_class, 
                                                     sel_chs=CONSTANT['OpenBMI']['sel_chs'])