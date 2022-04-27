import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm


class FEA_Net_h():
    # NOTICE: right now for homogeneous anisotropic material only!!
    def __init__(self, data, cfg):
        # set learning rate
        self.lr = cfg['lr']
        self.num_epoch = cfg['epoch']
        # self.batch_size = 4

        # data related
        self.num_node = data['num_node']
        self.E, self.mu, self.k, self.alpha = self.rho = data['rho'] #

        # 3 dimensional in and out, defined on the nodes
        self.load_pl = tf.placeholder(tf.float32, shape=(None, data['num_node'], data['num_node'], 3), name='load_pl')
        self.resp_pl = tf.placeholder(tf.float32, shape=(None, data['num_node'], data['num_node'], 3), name='resp_pl')

        # get filters
        self.get_w_matrix()
        self.load_pred = self.u2v_map()


    def get_w_matrix(self):
        self.get_w_matrix_elast()
        self.get_w_matrix_thermal()
        self.get_w_matrix_coupling()
        self.apply_physics_constrain()

    def apply_physics_constrain(self):
        # known physics
        self.wxx_tf = tf.constant(self.wxx_ref)
        self.wyy_tf = tf.constant(self.wyy_ref)
        self.wxy_tf = tf.constant(self.wxy_ref)
        self.wyx_tf = tf.constant(self.wyx_ref)
        self.wtt_tf = tf.constant(self.wtt_ref)
        self.wtx_tf = tf.constant(self.wtx_ref)
        self.wty_tf = tf.constant(self.wty_ref)

        # unknown physics
        self.wxt_np = np.zeros_like(self.wxt_ref)  # * 1.9
        self.wyt_np = np.zeros_like(self.wyt_ref)  # * 1.9

        # TF variable vector
        self.trainable_var_np = np.concatenate([self.wxt_np.flatten(),
                                                self.wyt_np.flatten()], 0)
        self.trainable_var_ref = np.concatenate([self.wxt_ref.flatten(),
                                                self.wyt_ref.flatten()], 0)
        self.trainable_var_pl = tf.placeholder(tf.float32, shape=(9 * 2,), name='filter_vector')

        wxt_tf, wyt_tf = tf.split(self.trainable_var_pl, 2)
        self.wxt_tf = tf.reshape(wxt_tf, (3, 3, 1, 1))
        self.wyt_tf = tf.reshape(wyt_tf, (3, 3, 1, 1))

        # add constrains
        self.singula_penalty = (tf.reduce_sum(self.wxt_tf)
                              + tf.reduce_sum(self.wyt_tf)
                                )**2
        # self.E = tf.clip_by_value(self.E, 0, 1)
        # self.mu = tf.clip_by_value(self.mu, 0, 0.5)

        # tf.nn.conv2d filter shape: [filter_height, filter_width, in_channels, out_channels]
        self.w_filter = tf.concat([tf.concat([self.wxx_tf, self.wxy_tf, self.wxt_tf], 2),
                                   tf.concat([self.wyx_tf, self.wyy_tf, self.wyt_tf], 2),
                                   tf.concat([self.wtx_tf, self.wty_tf, self.wtt_tf], 2)],
                                  3)

        self.w_filter_ref = np.concatenate([np.concatenate([self.wxx_ref, self.wxy_ref, self.wxt_ref], 2),
                                   np.concatenate([self.wyx_ref, self.wyy_ref, self.wyt_ref], 2),
                                   np.concatenate([self.wtx_ref, self.wty_ref, self.wtt_ref], 2)],
                                  3)

    def get_w_matrix_coupling(self):
        E, v = self.E, self.mu
        alpha = self.alpha
        self.wtx_ref = np.zeros((3,3,1,1), dtype='float32')
        self.wty_ref = np.zeros((3,3,1,1), dtype='float32')
        coef = E * alpha / (6*(v-1)) / 400 *1e6
        self.wxt_ref = coef * np.asarray([[1, 0, -1],
                                      [4, 0, -4],
                                      [1, 0, -1]]
                                     , dtype='float32').reshape(3,3,1,1)

        self.wyt_ref = coef * np.asarray([[-1, -4, -1],
                                      [0, 0, 0],
                                      [1, 4, 1]]
                                     , dtype='float32').reshape(3,3,1,1)

    def get_w_matrix_thermal(self):
        w = -1/3. * self.k * np.asarray([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
        w = np.asarray(w, dtype='float32')
        self.wtt_ref = w.reshape(3,3,1,1)

    def get_w_matrix_elast(self):
        E, mu = self.E, self.mu
        cost_coef = E / 16. / (1 - mu ** 2)
        wxx = cost_coef * np.asarray([
            [-4 * (1 - mu / 3.), 16 * mu / 3., -4 * (1 - mu / 3.)],
            [-8 * (1 + mu / 3.), 32. * (1 - mu / 3.), -8 * (1 + mu / 3.)],
            [-4 * (1 - mu / 3.), 16 * mu / 3., -4 * (1 - mu / 3.)],
        ], dtype='float32')

        wxy = wyx = cost_coef * np.asarray([
            [2 * (mu + 1), 0, -2 * (mu + 1)],
            [0, 0, 0],
            [-2 * (mu + 1), 0, 2 * (mu + 1)],
        ], dtype='float32')

        wyy = cost_coef * np.asarray([
            [-4 * (1 - mu / 3.), -8 * (1 + mu / 3.), -4 * (1 - mu / 3.)],
            [16 * mu / 3., 32. * (1 - mu / 3.), 16 * mu / 3.],
            [-4 * (1 - mu / 3.), -8 * (1 + mu / 3.), -4 * (1 - mu / 3.)],
        ], dtype='float32')

        self.wxx_ref = wxx.reshape(3,3,1,1)
        self.wxy_ref = wxy.reshape(3,3,1,1)
        self.wyx_ref = wyx.reshape(3,3,1,1)
        self.wyy_ref = wyy.reshape(3,3,1,1)

    def boundary_padding(self,x):
        ''' special symmetric boundary padding '''
        left = x[:, :, 1:2, :]
        right = x[:, :, -2:-1, :]
        upper = tf.concat([x[:, 1:2, 1:2, :], x[:, 1:2, :, :], x[:, 1:2, -2:-1, :]], 2)
        down = tf.concat([x[:, -2:-1, 1:2, :], x[:, -2:-1, :, :], x[:, -2:-1, -2:-1, :]], 2)
        padded_x = tf.concat([left, x, right], 2)
        padded_x = tf.concat([upper, padded_x, down], 1)
        return padded_x

    def u2v_map(self):
        padded_resp = self.boundary_padding(self.resp_pl)  # for boundary consideration
        wx = tf.nn.conv2d(input=padded_resp, filter=self.w_filter, strides=[1, 1, 1, 1], padding='VALID')
        return wx

    def get_loss(self):
        self.diff = self.load_pred - self.load_pl
        diff_not_on_bc = self.apply_bc(self.diff)
        self.l1_error = tf.reduce_mean(diff_not_on_bc**2)
        # self.l1_error = tf.reduce_mean((self.diff_not_on_bc*self.resp_pl[:,1:-1,1:-1,:])**2)
        self.loss = self.l1_error #+ self.singula_penalty
        return self.loss

    def get_grad(self):
        self.rho_grads = tf.gradients(self.loss, self.trainable_var_pl)
        return self.rho_grads

    def get_hessian(self):
        self.rho_hessian = tf.hessians(self.loss, self.trainable_var_pl)
        return self.rho_hessian

    # V2U mapping functions
    def apply_bc(self, x):
        x_bc = tf.pad(x[:, 1:-1, 1:-1, :], ((0,0), (1, 1),(1, 1), (0, 0)), "constant")  # for boundary consideration
        return x_bc

    def FEA_conv(self, w, x):
        padded_input = self.boundary_padding(x)  # for boundary consideration
        wx = tf.nn.conv2d(input=padded_input, filter=w, strides=[1, 1, 1, 1], padding='VALID')
        wx_bc = wx * self.bc_mask # boundary_corrrect
        return wx_bc

    def v2u_layer(self, w, x):
        wx = self.FEA_conv(w, x)
        wx_bc = self.apply_bc(wx)
        return wx_bc

    def get_dmat(self):
        d_matrix = tf.stack([self.wxx_tf[1,1,0,0], self.wyy_tf[1,1,0,0], self.wtt_tf[1,1,0,0]])  # x, y, and t components
        return tf.reshape(d_matrix,(1,1,1,3))

    def get_bc_mask(self):
        bc_mask = np.ones_like(self.new_load)
        bc_mask[:, 0, :, :] /= 2
        bc_mask[:, -1, :, :] /= 2
        bc_mask[:, :, 0, :] /= 2
        bc_mask[:, :, -1, :] /= 2
        return bc_mask

    def init_solve(self, load, omega):
        self.omega = omega
        self.new_load = load
        self.d_matrix = self.get_dmat()
        self.bc_mask = self.get_bc_mask()
        self.u_in = tf.placeholder(tf.float32, load.shape, name='u_in')
        self.u_out = self.apply(self.u_in)

    def apply(self, u_in):
        wx = self.v2u_layer(self.w_filter, u_in)
        u_out = self.omega * (self.new_load - wx) / self.d_matrix +  u_in
        return u_out