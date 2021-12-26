import numpy as np
from scipy import linalg 
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from min2net.utils import butter_bandpass_filter
# Note that we modify CSP function using MNE-Python package (version 0.20).
# Reference: 
# - https://github.com/mne-tools/mne-python
# - A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, R. Goj, M. Jas, T. Brooks, L. Parkkonen, and M. Ha ̈ma ̈la ̈inen, “Meg and eeg data analysis with mne-python,” Frontiers in Neuroscience, vol. 7, p. 267, 2013.
# Sources:
# - Common Spatial Pattern revisited by Riemannian geometry
# - Model based generalization analysis of common spatial pattern in brain computer interfaces

class SpectralSpatialMapping():
    def __init__(self, 
                bands,
                smp_freq,
                num_class=2, 
                order=5, 
                n_components=10, 
                n_pick_bands=20):
        self.bands = bands
        self.smp_freq = smp_freq
        self.n_components = n_components
        self.n_pick_bands = n_pick_bands
        self.cov_type = 'epoch'
        self.classes = None
        self.order = order
        self.n_classes = num_class
        if self.n_classes == 2:
            self.component_order = 'alternate'
        else:
            self.component_order = 'mutual_info'
        self.norm_trace = False
        
    def __covariance(self, X):
        # covariance = np.dot(X, X.T)/X.shape[0]
        # return covariance
        return np.dot(X,X.T)/np.trace(np.dot(X,X.T))
    
    # spatialFilter returns the spatial filter for mean covariance matrices on each class 
    def __get_spatial_filter(self, covs, sample_weights):
        eigen_vectors, eigen_values = self.__decompose_covs(covs, sample_weights)
        # CSP requires the eigenvalues and eigenvector be sorted in descending order
        sorted_index = self.__order_components(covs, sample_weights, eigen_vectors, eigen_values) 
        eigen_vectors = eigen_vectors[:, sorted_index]
        sp_filters = eigen_vectors.T
        pick_filters = sp_filters[:self.n_components]
        return pick_filters, sp_filters
    
    def __decompose_covs(self, covs, sample_weights):
        n_classes = len(covs)
        if n_classes == 2:
            eigen_values, eigen_vectors = linalg.eigh(covs[0], covs.sum(0))
        else:
            # The multiclass case is adapted from
            # http://github.com/alexandrebarachant/pyRiemann
            eigen_vectors, D = self.__ajd_pham(covs)
            eigen_vectors = self.__normalize_eigenvectors(eigen_vectors.T, covs, sample_weights)
            eigen_values = None
        return eigen_vectors, eigen_values
    
    def __calculate_covariance_matrices(self, data, y_class):
        '''The data is in the form of samples x channels x sampled_time_points'''
        if len(data.shape) != 3:
            raise Exception('Dimension is not match!')
        n_samples, n_channels, n_points = data.shape    
        
        if self.cov_type == 'concat':
            cov_estimator = self.__concat_cov
        elif self.cov_type == 'epoch':
            cov_estimator = self.__epoch_cov
            
        covs = []
        sample_weights = []
        self.classes = np.unique(y_class)
        n_classes = len(self.classes)
        # print('The numbers of class used are {} and type of class {}'.format(n_classes, self.classes))
        for id_class in self.classes:
            cov, weight = cov_estimator(data[y_class == id_class])
            if self.norm_trace:
                cov /= np.trace(cov)
            covs.append(cov)
            sample_weights.append(weight)
        return np.stack(covs), np.array(sample_weights)
    
    def __concat_cov(self, X_class):
        '''The data is in the form of samples x channels x sampled_time_points'''
        '''Concatenate epochs before computing the covariance.'''
        n_samples, n_channels, n_points = X_class.shape
        X_class = np.transpose(X_class, [1, 0, 2])
        X_class = X_class.reshape(n_channels, -1)
        cov = self.__covariance(X_class)
        # The covariace with norm trace is used to provide better results
        # cov = np.cov(X_class)
        weight = X_class.shape[0]
        return cov, weight

    def __epoch_cov(self, X_class):
        '''The data is in the form of samples x channels x sampled_time_points'''
        '''Mean of per-epoch covariances.'''
        cov = sum(self.__covariance(data) for data in X_class)
        cov /= len(X_class)
        weight = len(X_class)
        return cov, weight
    
    def __compute_mutual_info(self, covs, sample_weights, eigen_vectors):
        class_probas = sample_weights / sample_weights.sum()

        mutual_info = []
        for jj in range(eigen_vectors.shape[1]):
            aa, bb = 0, 0
            for (cov, prob) in zip(covs, class_probas):
                tmp = np.dot(np.dot(eigen_vectors[:, jj].T, cov),
                             eigen_vectors[:, jj])
                aa += prob * np.log(np.sqrt(tmp))
                bb += prob * (tmp ** 2 - 1)
            mi = - (aa + (3.0 / 16) * (bb ** 2))
            mutual_info.append(mi)

        return mutual_info

    def __normalize_eigenvectors(self, eigen_vectors, covs, sample_weights):
        # Here we apply an euclidean mean. See pyRiemann for other metrics
        mean_cov = np.average(covs, axis=0, weights=sample_weights)

        for ii in range(eigen_vectors.shape[1]):
            tmp = np.dot(np.dot(eigen_vectors[:, ii].T, mean_cov),
                         eigen_vectors[:, ii])
            eigen_vectors[:, ii] /= np.sqrt(tmp)
        return eigen_vectors
    
    def __order_components(self, covs, sample_weights, eigen_vectors, eigen_values):
        if self.component_order == 'mutual_info' and self.n_classes > 2:
            mutual_info = self.__compute_mutual_info(covs, sample_weights, eigen_vectors)
            ix = np.argsort(mutual_info)[::-1]
        elif self.component_order == 'mutual_info' and self.n_classes == 2:
            ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        elif self.component_order == 'alternate' and self.n_classes == 2:
            i = np.argsort(eigen_values)
            ix = np.empty_like(i)
            ix[1::2] = i[:len(i) // 2]
            ix[0::2] = i[len(i) // 2:][::-1]
        return ix
    
    def __ajd_pham(self, X, eps=1e-6, max_iter=15):
        '''Approximate joint diagonalization based on Pham's algorithm.
        This is a direct implementation of the PHAM's AJD algorithm [1].
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_channels)
            A set of covariance matrices to diagonalize.
        eps : float, default 1e-6
            The tolerance for stopping criterion.
        max_iter : int, default 1000
            The maximum number of iteration to reach convergence.
        Returns
        -------
        V : ndarray, shape (n_channels, n_channels)
            The diagonalizer.
        D : ndarray, shape (n_epochs, n_channels, n_channels)
            The set of quasi diagonal matrices.
        References
        ----------
        .. [1] Pham, Dinh Tuan. 'Joint approximate diagonalization of positive
               definite Hermitian matrices.' SIAM Journal on Matrix Analysis and
               Applications 22, no. 4 (2001): 1136-1152.
        '''
        # Adapted from http://github.com/alexandrebarachant/pyRiemann
        n_epochs = X.shape[0]

        # Reshape input matrix
        A = np.concatenate(X, axis=0).T

        # Init variables
        n_times, n_m = A.shape
        V = np.eye(n_times)
        epsilon = n_times * (n_times - 1) * eps

        for it in range(max_iter):
            decr = 0
            for ii in range(1, n_times):
                for jj in range(ii):
                    Ii = np.arange(ii, n_m, n_times)
                    Ij = np.arange(jj, n_m, n_times)

                    c1 = A[ii, Ii]
                    c2 = A[jj, Ij]

                    g12 = np.mean(A[ii, Ij] / c1)
                    g21 = np.mean(A[ii, Ij] / c2)

                    omega21 = np.mean(c1 / c2)
                    omega12 = np.mean(c2 / c1)
                    omega = np.sqrt(omega12 * omega21)

                    tmp = np.sqrt(omega21 / omega12)
                    tmp1 = (tmp * g12 + g21) / (omega + 1)
                    tmp2 = (tmp * g12 - g21) / max(omega - 1, 1e-9)

                    h12 = tmp1 + tmp2
                    h21 = np.conj((tmp1 - tmp2) / tmp)

                    decr += n_epochs * (g12 * np.conj(h12) + g21 * h21) / 2.0

                    tmp = 1 + 1.j * 0.5 * np.imag(h12 * h21)
                    tmp = np.real(tmp + np.sqrt(tmp ** 2 - h12 * h21))
                    tau = np.array([[1, -h12 / tmp], [-h21 / tmp, 1]])

                    A[[ii, jj], :] = np.dot(tau, A[[ii, jj], :])
                    tmp = np.c_[A[:, Ii], A[:, Ij]]
                    tmp = np.reshape(tmp, (n_times * n_epochs, 2), order='F')
                    tmp = np.dot(tmp, tau.T)

                    tmp = np.reshape(tmp, (n_times, n_epochs * 2), order='F')
                    A[:, Ii] = tmp[:, :n_epochs]
                    A[:, Ij] = tmp[:, n_epochs:]
                    V[[ii, jj], :] = np.dot(tau, V[[ii, jj], :])
            if decr < epsilon:
                break
        D = np.reshape(A, (n_times, -1, n_times)).transpose(1, 0, 2)
        return V, D

    def __get_log_var_feats(self, spatial_filt, data):
        data_dot = np.dot(spatial_filt, data) # spatially filtered signals 
        data_var = np.var(data_dot)
        # We use log(var) instead of averaging
        data_log = np.log(data_var)
        return data_log

    def get_transformed_feats(self, spatial_filt, data):
        data_dot = np.dot(spatial_filt, data) # spatially filtered signals 
        data_cov = self.__covariance(data_dot)
        return data_cov

    def spatial_spectral_with_valset(self, X_tr, y_tr, X_val, X_te):  
        if len(X_tr.shape) != 3:
            raise Exception('Dimension is not match!')
            
        # Empty array for getting the transformed features in each frequency band    
        X_tr_transformed_var = np.zeros((len(self.bands), X_tr.shape[0]))
        
        # Empty array for getting the covariance matrix 
        X_tr_transformed_cov = np.zeros((len(self.bands), X_tr.shape[0], X_tr.shape[1], X_tr.shape[1]))
        X_val_transformed_cov = np.zeros((len(self.bands), X_val.shape[0], X_val.shape[1], X_val.shape[1]))
        X_te_transformed_cov = np.zeros((len(self.bands), X_te.shape[0], X_te.shape[1], X_te.shape[1]))

        for id_band, freq_band in enumerate(self.bands):
            print('The process is holding on ID {} and Freq Interval {}'.format(id_band, freq_band))

            X_tr_filtered = butter_bandpass_filter(X_tr, freq_band[0], freq_band[1], self.smp_freq, self.order)
            X_val_filtered = butter_bandpass_filter(X_val, freq_band[0], freq_band[1], self.smp_freq, self.order)
            X_te_filtered = butter_bandpass_filter(X_te, freq_band[0], freq_band[1], self.smp_freq, self.order)

            # Calculating covariance only on training set
            covs, sample_weights =  self.__calculate_covariance_matrices(X_tr_filtered, y_tr)
            spf_sel, spf_org = self.__get_spatial_filter(covs, sample_weights)

            # Calculate the variance of spatially filtered signals and then compute the logarithm in each single EEG trial
            for sample_tr in range(X_tr_filtered.shape[0]):
                X_tr_transformed_var[id_band, sample_tr] = self.__get_log_var_feats(spf_sel, X_tr_filtered[sample_tr,:,:])  
                X_tr_transformed_cov[id_band, sample_tr,:,:] = self.get_transformed_feats(spf_org, X_tr_filtered[sample_tr,:,:]) 

            for sample_val in range(X_val_filtered.shape[0]):
                X_val_transformed_cov[id_band, sample_val,:,:] = self.get_transformed_feats(spf_org, X_val_filtered[sample_val,:,:])

            for sample_te in range(X_te_filtered.shape[0]):
                X_te_transformed_cov[id_band, sample_te,:,:] = self.get_transformed_feats(spf_org, X_te_filtered[sample_te,:,:]) 

        # Prepare the most suited form of data for rearanging all frequency bands
        X_tr_transformed_var_sw = np.swapaxes(X_tr_transformed_var,0,1) 

        # Rearrange all frequency bands from the largest MI value to smallest MI value and pick k frequency bands by considering the most k largest MI values.
        # This procedure considered only traning set obtaning the order of k frequnecy bands that provide the largest MI value.
        selector = SelectKBest(score_func=mutual_info_classif, k=self.n_pick_bands)
        X_tr_picked_band = selector.fit_transform(X_tr_transformed_var_sw, y_tr)
        des_order_list = selector.get_support(indices=True)

        # Display the selected subbands
        for idx_band in des_order_list:
            print('band_name:', self.bands[idx_band] ,'---', 'Index_is:', idx_band)

        X_tr_se_band = np.take(np.swapaxes(X_tr_transformed_cov,0,1), des_order_list, axis=1)
        X_val_se_band = np.take(np.swapaxes(X_val_transformed_cov,0,1), des_order_list, axis=1)
        X_te_se_band = np.take(np.swapaxes(X_te_transformed_cov,0,1), des_order_list, axis=1)
        # All data are in the form of (n_samples,n_subbands,n_components,n_components)
        return X_tr_se_band, X_val_se_band, X_te_se_band 
