import numpy as np
import scipy.io as sio
#import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
#from FEA_Net import *

class Evaluator(object):
    def __init__(self, model, data):
        self.model = model

        self.data = data
        self.init_w = np.zeros((3,3,1,1))

        self.loss_value = None
        self.grads_value = None

        self.loss_tf = self.model.get_loss()
        self.hessian_tf = self.model.get_hessian()
        self.grad_tf = self.model.get_grad()
        self.initial_graph()

    def initial_graph(self):
        # initialize
        FLAGS = tf.app.flags.FLAGS
        tfconfig = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
        )
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def get_loss(self, w):
        self.feed_dict = {self.model.load_pl: data['train_load'],
                          self.model.resp_pl: data['train_resp'],
                          self.model.trainable_var_pl: w}
        self.loss_value = self.sess.run(self.loss_tf, self.feed_dict).astype('float64')
        return self.loss_value

    def get_grads(self, w):
        self.feed_dict = {self.model.load_pl: data['train_load'],
                          self.model.resp_pl: data['train_resp'],
                          self.model.trainable_var_pl: w}
        self.grads_value = self.sess.run(self.grad_tf, self.feed_dict)[0].flatten().astype('float64')
        return self.grads_value

    def get_hessian(self, w):
        self.feed_dict = {self.model.load_pl: data['train_load'],
                          self.model.resp_pl: data['train_resp'],
                          self.model.trainable_var_pl: w}
        self.hessian_value = self.sess.run(self.hessian_tf, self.feed_dict)[0].astype('float64')
        return self.hessian_value

    def get_pred(self,w):
        feed_dict = {self.model.load_pl: data['train_load'],
                      self.model.resp_pl: data['train_resp'],
                      self.model.trainable_var_pl: w.astype('float32')}
        pred_value = self.sess.run(self.model.load_pred, feed_dict)
        return pred_value

    def run_BFGS(self):
        from scipy.optimize import fmin_l_bfgs_b
        x, min_val, info = fmin_l_bfgs_b(self.get_loss, self.init_w.flatten(),
                                         fprime=self.get_grads, maxiter=200, maxfun=200,
                                         disp= True)
        print('    loss: {}'.format(min_val))
        pass

    def run_newton(self):
        from scipy.optimize import minimize
        self.result = minimize(self.get_loss, self.model.trainable_var_np, method='Newton-CG',
                          jac=self.get_grads, hess=self.get_hessian,
                          options={'xtol': 1e-8, 'disp': True})
        return self.result.x

    def run_trust_ncg(self):
        from scipy.optimize import minimize
        self.result = minimize(self.get_loss, self.model.trainable_var_np, method='trust-ncg',
                          jac=self.get_grads, hess=self.get_hessian,
                          options={'gtol': 1e-3, 'disp': True})
        return self.result.x

    def run_tnc(self):
        from scipy.optimize import fmin_tnc

        self.result = fmin_tnc(self.get_loss,
                               self.model.trainable_var_np,
                               fprime=self.get_grads,
                               stepmx=100,
                               pgtol=1e-5,
                               # ftol=1e-15,
                               maxfun=20000,
                               disp='True')

        return self.result[0]

    def run_least_squares(self):
        # problematic
        from scipy.optimize import least_squares
        self.result = least_squares(self.get_loss,
                               self.model.trainable_var_np,
                               jac=self.get_grads,
                               method='trf',
                               gtol=1e-5,
                               # ftol=1e-15,
                               loss='linear',
                               verbose=2)

        return self.result[0]

    def visualize(self, w):
        pred_value = self.get_pred(w)
        plt.figure(figsize=(6, 6))
        idx = 0  # which data to visualize
        for i in range(3):
            plt.subplot(4, 3, i + 1)
            plt.imshow(self.data['test_resp'][idx, 1:-1, 1:-1, i])
            plt.colorbar()
            plt.subplot(4, 3, 3 + i + 1)
            plt.imshow(self.data['test_load'][idx, 1:-1, 1:-1, i])
            plt.colorbar()
            plt.subplot(4, 3, 6 + i + 1)
            plt.imshow(pred_value[idx, 1:-1, 1:-1, i])
            plt.colorbar()
            plt.subplot(4, 3, 9 + i + 1)
            plt.imshow(self.data['test_load'][idx, 1:-1, 1:-1, i] - pred_value[idx, 1:-1, 1:-1, i])
            plt.colorbar()
        plt.show()

    def init_solve(self, load, omega=2/3.):
        self.model.init_solve(load, omega)
        self.solution = {'itr':[], 'loss': [], 'pred':[]}

    def run_forward(self, filter, pred_i, resp_ref=None, max_itr=100):

        st = 0 if self.solution['itr'] == [] else self.solution['itr'][-1]+10
        for itr in tqdm(range(st, st+max_itr, 1)):
            feed_dict = {self.model.u_in: pred_i, self.model.trainable_var_pl:filter}
            pred_i = self.sess.run(self.model.u_out, feed_dict)
            if itr%1 == 0:
                self.solution['itr'] += [itr]
                self.solution['pred'] += [pred_i]
                if resp_ref is not None:
                    pred_err_i = np.sqrt(np.sum((resp_ref - pred_i) ** 2)) / np.sqrt(  np.sum((resp_ref) ** 2))
                    print("iter:{}  pred_err: {}".format(itr, np.mean(pred_err_i)))
                    self.solution['loss'] += [np.mean(pred_err_i)]

        return pred_i
