import numpy as npfrom data.data_generator import data_generatordef sanity_checks(x, theta, t_max=100, dt=1, to_tensor=True, threshold_S=0.25):    """    Implement sanity checks    :param X:    :param theta:    :param t:    :param initial_values:    :return:    """    N = x.shape[0]    num_instances = 0    while num_instances != N:        if N - num_instances != N:            extra_data = data_generator(batch_size=(N - num_instances), t_max=t_max, dt=dt, N=N, to_tensor=to_tensor)            extra_X = np.array(extra_data['x'])            extra_theta = np.array(extra_data['theta'])            x = np.concatenate((x, extra_X))            theta = np.concatenate((theta, extra_theta))        idx_S = np.argwhere(x[:, -1, 0] > threshold_S)        idx = np.unique(idx_S)        x = np.delete(x, idx, axis=0)        theta = np.delete(theta, idx, axis=0)        num_instances = x.shape[0]    # x = x.reshape(x.shape[0], -1)    return x, thetadef est_sanity_checks(theta_approx_means, theta_test, x_test=None, idx_mask=False):    # parameters = ['beta', 'alpha', 'gamma', 'delta', 'rho']    low = [0.8, 0.25, 0.1, 0.01, 0.1]    high = [2.25, 0.75, 1.0, 0.4, 0.6]    # array-like of shape (n_tuples, 5)    mask = [(theta_approx_means[:, i] > low[i]) * (theta_approx_means[:, i] < high[i]) for i in range(len(low))]    mask = np.array(mask).T    ftheta_approx_means = np.delete(theta_approx_means, np.argwhere(mask == False)[:, 0], axis=0)    ftheta_test = np.delete(theta_test, np.argwhere(mask == False)[:, 0], axis=0)    assert ftheta_approx_means.shape == ftheta_test.shape    if x_test is not None:        fx_test = np.delete(x_test, np.argwhere(mask == False)[:, 0], axis=0)        assert fx_test.shape[0] == ftheta_approx_means.shape[0]        if idx_mask:            mask_idx = np.argwhere(mask == False)[:, 0]            return ftheta_approx_means, ftheta_test, fx_test, mask_idx        else:            return ftheta_approx_means, ftheta_test, fx_test    else:        if idx_mask:            mask_idx = np.argwhere(mask == False)[:, 0]            return ftheta_approx_means, ftheta_test, mask_idx        else:            return ftheta_approx_means, ftheta_test