import numpy as np
import scipy.io as sio


def load_FEANet_data(snr=None, percent=None):
    num_node = 37

    # coupled loading
    data = sio.loadmat('data/thermoelastic/2D_thermoelastic_36by36_xy_fixed_single_data_half_loading.mat')

    load = np.expand_dims(np.stack([-data['fx'], -data['fy'], data['ftem']], -1), 0).astype('float32')
    resp = np.expand_dims(np.stack([data['ux']*1e6, data['uy']*1e6, data['utem']], -1), 0).astype('float32')

    if percent is not None:
        loading_w_noise = np.zeros_like(load)
        response_w_noise = np.zeros_like(resp)
        for i in range(load.shape[0]):
            for j in range(load.shape[-1]):
                noise = percent * np.random.normal(size=load.shape)
                loading_w_noise = (1+noise) * load
                noise = percent * np.random.normal(size=load.shape)
                response_w_noise = (1+noise) * resp
    elif snr is not None:
        loading_w_noise = np.zeros_like(load)
        response_w_noise = np.zeros_like(resp)
        for i in range(load.shape[0]):
            for j in range(load.shape[-1]):
                low_val = load.min() / 10 ** (snr / 20)
                max_val = load.max() / 10 ** (snr / 20)
                noise = np.random.uniform(low=low_val, high=max_val, size=(load.shape[1:3]))
                loading_w_noise[i, :, :, j] = noise + load[i, :, :, j]
                low_val = resp.min() / 10 ** (snr / 20)
                max_val = resp.max() / 10 ** (snr / 20)
                noise = np.random.uniform(low=low_val, high=max_val, size=(resp.shape[1:3]))
                response_w_noise[i, :, :, j] = noise + resp[i, :, :, j]
        # resp = response_w_noise
        # load = loading_w_noise

    rho = [212e3, 0.288, 16., 12e-6] # E, mu, k, alpha

    train_load = loading_w_noise
    train_resp = response_w_noise
    test_load = load
    test_resp = resp
    data = {'num_node': num_node,
            'rho': rho,
            'train_load': train_load,
            'train_resp': train_resp,
            'test_load': test_load,
            'test_resp': test_resp,
            }

    return data
