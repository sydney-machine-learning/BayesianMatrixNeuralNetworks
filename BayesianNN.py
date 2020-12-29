import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import pickle


class BayesianRegressorWrapper(nn.Module):
    """
    Bayesian wrapper for a regression network model.
    Assumes Gaussian likelihood, Gaussian priors for weights and biases and an Inverse Gamma prior for tausq
    """

    def __init__(self, base_model, sigmasq=25, nu1=1.0, nu2=1.0):
        """
        Parameters:
        ----------
        base_model: torch.module
            A pytorch neural network

        sigmasq: float, optional
            Prior variance for weights, biases

        nu1, nu2: float, optional
            Prior parameters for tausq
        """
        super().__init__()
        self.base_model = base_model
        self.sigmasq, self.nu1, self.nu2 = sigmasq, nu1, nu2
        self.tausq = None
        self.param_dist = []

    def compute_log_posterior(self, X, y, w=None, tausq=None):
        if w is not None:
            w_cur = deepcopy(list(self.parameters()))
            tausq_cur = self.tausq
            self.set_params(w, tausq)

        y_pred = self.forward(X)

        n = len(X)
        part1 = -0.5 * n * np.log(self.tausq) - 0.5 * torch.sum((y - y_pred) ** 2) / self.tausq
        part2 = -(0.5 / self.sigmasq) * sum([torch.sum(w_ ** 2) for w_ in list(self.parameters())])
        part3 = -2 * (self.nu1 + 1) * np.log(self.tausq) - self.nu2 / self.tausq

        if w is not None:
            self.set_params(w_cur, tausq_cur)

        return part1 + part2 + part3

    def set_params(self, new_w, new_tausq=None):
        for w, new_w in zip(self.parameters(), new_w):
            w.data = new_w.data

        if new_tausq is not None:
            self.tausq = new_tausq

    def update_param_dist(self, param_dist, extend=True):
        """Updates list of weights and biases. """
        if extend:
            self.param_dist.extend(param_dist)
        else:
            self.param_dist = param_dist

    def compute_pred_dist(self, i):
        """Computes multiple predictions using param_dist. """
        pred_list = []
        w_cur = deepcopy(list(self.parameters()))
        for w_ in self.param_dist:
            self.set_params(w_)
            pred_list.append(self(i))
        self.set_params(w_cur)

        return pred_list

    def forward(self, i):
        return self.base_model.forward(i)

    def parameters(self):
        return self.base_model.parameters()


class RegressionSampler(torch.optim.Optimizer):
    """
    MCMC sampler for a Bayesian regression network.
    Supports either random walk MCMC or Langevin gradient MCMC
    """
    def __init__(self, bayesian_model,
                 X_train, y_train, X_test=None, y_test=None,
                 use_lg=True, lg_freq=0.30, w_prop_sd=0.01, tausq_prop_sd=0.01, lr=0.01):
        """
        Parameters:
        ----------
        bayesian_model: BayesianRegressorWrapper
            A Bayesian regression network model

        X_train, y_train: torch.tensor
            Training data inputted as a pytorch tensor

        X_test, y_test: torch.tensor, optional
            Test data, needs to be inputted if test loss is to be computed during training

        use_lg: bool, optional
            Flag indicating if Langevin gradients are to be used

        lg_freq: float, optional
            Proportion of the time Langevin gradients are to be used

        w_prop_sd: float, optional
            Standard deviation of the proposal distribution for weight and bias updates

        tausq_prop_sd: float, optional
            Standard deviation of the proposal distribution for tausq

        lr: float, optional
            Learning rate to be used for Langevin gradient updates
        """
        self.model = bayesian_model
        self.update_data(X_train, y_train, X_test, y_test)

        self.use_lg = use_lg
        self.lg_freq = lg_freq
        self.lr = lr
        self.w_prop_sd = w_prop_sd
        self.tausq_prop_sd = tausq_prop_sd
        self.criterion = nn.MSELoss()

        super().__init__(self.model.parameters(), {})
        self.chain_initialiser()

    def sample(self, evaluate_on_test=False):
        """
        Main method of the class: draws a single MCMC sample, computes train predictions and losses
        and optionally test predictions and losses.
        """
        tausq_p = np.exp(np.log(self.model.tausq) + np.random.normal(scale=self.tausq_prop_sd))

        if self.use_lg:
            if np.random.rand() < self.lg_freq:
                w_p = [nn.Parameter(
                    w_ + torch.normal(mean=torch.zeros_like(w_), std=torch.ones_like(w_) * self.w_prop_sd)
                ) for w_ in self.w_k_bar]
                y_pred_train_p, loss_p, w_p_bar = \
                    self.compute_loss_and_gradient_update(self.X_train, self.y_train, w=w_p)
                log_q_ratio = self.compute_log_q_ratio(w_p, w_p_bar, self.w_k, self.w_k_bar)
            else:
                w_p = [nn.Parameter(
                    w_ + torch.normal(mean=torch.zeros_like(w_), std=torch.ones_like(w_) * self.w_prop_sd)
                ) for w_ in self.w_k]
                y_pred_train_p, loss_p, w_p_bar = \
                    self.compute_loss_and_gradient_update(self.X_train, self.y_train, w_p)
                log_q_ratio = 0.0

        else:
            w_p = [nn.Parameter(
                w_ + torch.normal(mean=torch.zeros_like(w_), std=torch.ones_like(w_) * self.w_prop_sd)
            ) for w_ in self.w_k] # w_k_list[-1]
            y_pred_train_p, loss_p = \
                self.compute_loss_and_gradient_update(self.X_train, self.y_train, w=w_p, grad=False)
            log_q_ratio = 0.0

        alpha = torch.exp(
            self.model.compute_log_posterior(self.X_train, self.y_train, w_p, tausq_p)
            - self.model.compute_log_posterior(self.X_train, self.y_train)
            + log_q_ratio)

        if alpha > np.random.rand():
            self.accept_counter += 1
            self.model.set_params(w_p, tausq_p)
            self.w_k = w_p

            if self.use_lg:
                self.w_k_bar = w_p_bar

            if evaluate_on_test:
                y_pred_test, test_loss = \
                    self.compute_loss_and_gradient_update(self.X_test, self.y_test, grad=False)
                self.container.append(self.w_k, self.model.tausq, y_pred_train_p, loss_p, y_pred_test, test_loss)
            else:
                self.container.append(self.w_k, self.model.tausq, y_pred_train_p, loss_p)
        else:
            self.container.append_last()

    def compute_loss_and_gradient_update(self, X, y, w=None, grad=True):
        """
        Computes predictions, losses and optionally gradient updated weights and biases

        Parameters:
        -----------
        X: torch.tensor
            Features to predict from.

        y: torch.tensor
            Targets to calculate loss with.

        w: list, optional
            Parameters to compute predictions, if None, uses the base_model's current parameters.

        grad: bool, optional
            Flag indicating if gradient updated weights and biases should be computed. Set to False when RW-MCMC is
            used.
        """
        if w is not None:
            w_cur = deepcopy(list(self.model.parameters()))
            self.model.set_params(w)

        y_pred = self.model(X)
        loss = self.criterion(y_pred, y)

        if not grad:
            if w is not None:
                self.model.set_params(w_cur)
            return y_pred, loss

        self.zero_grad()
        loss.backward()
        nabla_w = [w_.grad if w_.grad is not None else torch.zeros_like(w_) for w_ in self.model.parameters()]
        with torch.no_grad():
            w_bar = [w_.add(grad_, alpha=-self.lr) for w_, grad_ in zip(list(self.model.parameters()), nabla_w)]

        if w is not None:
            self.model.set_params(w_cur)

        return y_pred, loss.item(), w_bar

    def compute_log_q_ratio(self, w_p, w_p_bar, w_k, w_k_bar):
        """Computes the logarithm q ratio, used to calculate the acceptance probability. """
        log_num = sum([torch.sum((w_p_bar_ - w_k_) ** 2) for w_p_bar_, w_k_ in zip(w_p_bar, w_k)])
        log_den = sum([torch.sum((w_p_ - w_k_bar_) ** 2) for w_p_, w_k_bar_ in zip(w_p, w_k_bar)])
        return -1 / (2 * self.w_prop_sd ** 2) * (log_num - log_den)

    def clear(self): # create data structure class to contain
        """Initialises or empties containers for samples. """
        self.container = SampleContainer()

    def update_data(self, X_train, y_train, X_test, y_test):
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

    def update_param_dist(self, w_list=None):
        """Updates param_dist of base_model. """
        if w_list is None:
            self.model.update_param_dist(self.container.w_list)
        else:
            self.model.update_param_dist(w_list)

    def chain_initialiser(self, new_w=None, new_tausq=None):
        """
        Initialises chain when the instance is first created or when parameters are updated.
        Clears previously obtained samples so make sure these are saved first if they are needed.
        """
        if new_w is not None:
            self.model.set_params(new_w, new_tausq)

        self.clear()
        self.accept_counter = 0

        self.w_k = list(self.model.parameters())

        if self.use_lg:
            y_pred_train, loss, self.w_k_bar = \
                self.compute_loss_and_gradient_update(self.X_train, self.y_train)
        else:
            y_pred_train, loss = \
                self.compute_loss_and_gradient_update(self.X_train, self.y_train, grad=False)

        if self.model.tausq is None:
            self.model.tausq = float(torch.var(self.model(self.X_train) - self.y_train))

        if self.X_test is None:
            self.container.append(self.w_k, self.model.tausq, y_pred_train, loss)
        else:
            y_pred_test, test_loss = \
                self.compute_loss_and_gradient_update(self.X_test, self.y_test, grad=False)
            self.container.append(self.w_k, self.model.tausq, y_pred_train, loss, y_pred_test, test_loss)

    def save_data(self, folder):
        self.container.save_data(folder)


class SampleContainer:
    def __init__(self):
        self.w_list = []
        self.tausq_list = []
        self.y_pred_train_list = []
        self.loss_list = []
        self.y_pred_test_list = []
        self.test_loss_list = []

    def append(self, w, tausq, y_pred_train, loss, y_pred_test=None, test_loss=None):
        self.w_list.append(w)
        self.tausq_list.append(tausq)
        self.y_pred_train_list.append(y_pred_train)
        self.loss_list.append(loss)
        self.y_pred_test_list.append(y_pred_test)
        self.test_loss_list.append(test_loss)

    def retrieve_last(self):
        return (self.w_list[-1], self.tausq_list[-1], self.y_pred_train_list[-1],
                self.loss_list[-1], self.y_pred_test_list[-1], self.test_loss_list[-1])

    def append_last(self):
        self.append(*self.retrieve_last())

    def save_data(self, folder):
        with open(f'{folder}/w_list_{j}.pickle', 'wb') as b:
            pickle.dump(self.w_list, b)

        with open(f'{folder}/tausq_list_{j}.pickle', 'wb') as b:
            pickle.dump(self.tausq_list, b)

        with open(f'{folder}/y_pred_train_list_{j}.pickle', 'wb') as b:
            pickle.dump(self.y_pred_train_list, b)

        with open(f'{folder}/loss_list_{j}.pickle', 'wb') as b:
            pickle.dump(self.loss_list, b)

        with open(f'{folder}/y_pred_test_list_{j}.pickle', 'wb') as b:
            pickle.dump(self.y_pred_test_list, b)

        with open(f'{folder}/test_loss_list){j}.pickle', 'wb') as b:
            pickle.dump(self.test_loss_list, b)