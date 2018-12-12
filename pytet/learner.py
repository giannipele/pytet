from autograd import grad
from autograd.misc.optimizers import adam
from sklearn.model_selection import train_test_split
import autograd.numpy as np
from utils import TetValue

class Learner():
    def __init__(self, tet, loss, x, y):
        self.tet = tet
        self.x = x
        self.y = y
        self.loss = loss
        self.best_params = []
        self.best_v_err = float("inf")

    def learn(self, **kwargs): 
        params = self.tet.get_params()
        optimizer = kwargs["optimizer"]

        objective_grad = grad(self.calculate_loss, argnum=0)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.x, self.y, test_size=0.1, random_state=42) 
        print("DATASET SIZE\tTrain set: {}\tValidation set: {}".format(len(self.X_train), len(self.X_val)))

        if optimizer == "adam":
            num_iters = kwargs["num_iters"]
            step_size = kwargs["step_size"]
            optimized_params = adam(objective_grad, params, step_size=step_size, num_iters=num_iters, callback=self.print_perf)
            print("BEST VALIDATION ERROR: ", self.best_v_err)
            print("BEST PARAMS: ", self.best_params)
            return optimized_params

    def calculate_loss(self, params, iteration):
        err = 0
        ys_hat = []
        log_ev = []
        for tetvalue in self.X_train:
            #ev_tree = TetValue()
            ys_hat.append(self.tet.forward_value(params, tetvalue))
            #log_ev.append(ev_tree)
        err = self.loss.loss(ys_hat, self.y_train)
        self.err = err
        return err


    def print_perf(self, params, it, gradient):
        preds = [(self.tet.forward_value(params, tetvalue) - target)**2 for tetvalue, target in zip(self.X_val, self.y_val)]
        v_err = np.sum(preds)/(2*len(preds))
        if v_err < self.best_v_err:
            self.best_v_err = v_err
            self.best_params = params
        print("{}\t|\t{}\t|\t{}\t|\t{}\t|\t{}".format(it, self.err._value, v_err, params, gradient))



