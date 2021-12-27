import min2net.preprocessing as prep
from min2net.preprocessing.config import CONSTANT

k_folds = 5
pick_smp_freq = 100
n_components = 10
bands = [[7.5,14],[11,13],[10,14],[9,12],[19,22],[16,22],[26,34],[17.5,20.5],[7,30],[5,14],[11,31],
         [12,18],[7,9],[15,17],[25,30],[20,25],[5,10],[10,25],[15,30],[10,12],[23,27],[28,32],[12,33],
         [11,22],[5,8],[7.5,17.5],[23,26],[5,20],[5,25],[10,20]]
n_pick_bands = 20
order = 5
save_path = 'datasets'
num_class = 2

prep.SMR_BCI.spectral_spatial.subject_dependent_setting(k_folds=k_folds, 
                                                        pick_smp_freq=pick_smp_freq,
                                                        n_components=n_components, 
                                                        bands=bands, 
                                                        n_pick_bands=n_pick_bands, 
                                                        order=order, 
                                                        save_path=save_path, 
                                                        num_class=num_class,
                                                        sel_chs=CONSTANT['SMR_BCI']['sel_chs'])

prep.BCIC2a.spectral_spatial.subject_dependent_setting(k_folds=k_folds, 
                                                       pick_smp_freq=pick_smp_freq,
                                                       n_components=n_components, 
                                                       bands=bands, 
                                                       n_pick_bands=n_pick_bands, 
                                                       order=order, 
                                                       save_path=save_path, 
                                                       num_class=num_class,
                                                       sel_chs=CONSTANT['BCIC2a']['sel_chs'])

prep.OpenBMI.spectral_spatial.subject_dependent_setting(k_folds=k_folds, 
                                                        pick_smp_freq=pick_smp_freq,
                                                        n_components=n_components, 
                                                        bands=bands, 
                                                        n_pick_bands=n_pick_bands, 
                                                        order=order, 
                                                        save_path=save_path, 
                                                        num_class=num_class,
                                                        sel_chs=CONSTANT['OpenBMI']['sel_chs'])

prep.SMR_BCI.spectral_spatial.subject_independent_setting(k_folds=k_folds, 
                                                          pick_smp_freq=pick_smp_freq,
                                                          n_components=n_components, 
                                                          bands=bands, 
                                                          n_pick_bands=n_pick_bands, 
                                                          order=order, 
                                                          save_path=save_path, 
                                                          num_class=num_class,
                                                          sel_chs=CONSTANT['SMR_BCI']['sel_chs'])

prep.BCIC2a.spectral_spatial.subject_independent_setting(k_folds=k_folds, 
                                                         pick_smp_freq=pick_smp_freq,
                                                         n_components=n_components, 
                                                         bands=bands, 
                                                         n_pick_bands=n_pick_bands, 
                                                         order=order, 
                                                         save_path=save_path, 
                                                         num_class=num_class,
                                                         sel_chs=CONSTANT['BCIC2a']['sel_chs'])

prep.OpenBMI.spectral_spatial.subject_independent_setting(k_folds=k_folds, 
                                                          pick_smp_freq=pick_smp_freq,
                                                          n_components=n_components, 
                                                          bands=bands, 
                                                          n_pick_bands=n_pick_bands, 
                                                          order=order, 
                                                          save_path=save_path, 
                                                          num_class=num_class,
                                                          sel_chs=CONSTANT['OpenBMI']['sel_chs'])