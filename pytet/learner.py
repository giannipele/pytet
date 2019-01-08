from autograd import grad
from autograd.misc.optimizers import adam
from sklearn.model_selection import train_test_split
from value import TetValue
import autograd.numpy as np
import random
import itertools


class Learner:
    def __init__(self, tet, loss, x, y):
        self.tet = tet
        self.X = X
        self.y = y
        self.loss = loss
        self.best_params = []
        self.best_v_err = float("inf")
        self.err = -1

    def learn(self, **kwargs): 

        params = self.tet.get_params()
        optimizer = kwargs["optimizer"]

        objective_grad = grad(self.calculate_loss, argnum=0)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.01, random_state=42)
        print("DATASET SIZE\tTrain set: {}\tValidation set: {}".format(len(self.X_train), len(self.X_val)))

        if optimizer == "adam":
            num_iters = kwargs["num_iters"]
            step_size = kwargs["step_size"]
            optimized_params = adam(objective_grad, params, step_size=step_size, num_iters=num_iters, callback=self.print_perf)
            print("BEST VALIDATION ERROR: ", self.best_v_err)
            print("BEST PARAMS: ", self.best_params)
            return optimized_params

    def calculate_loss(self, params, iteration):
        ys_hat = []
        evaluations = []
        for tetvalue in self.X_train:
            ev_tree = TetValue()
            ys_hat.append(self.tet.forward_value(params, tetvalue, ev_tree))
            evaluations.append(ev_tree)
        err = self.loss.loss(ys_hat, self.y_train)
        self.err = err
        return err

    def print_perf(self, params, it, gradient):
        predictions = [self.tet.forward_value(params, tetvalue) for tetvalue in self.X_val]
        err = self.loss.loss(predictions, self.y_val)
        if err < self.best_v_err:
            self.best_v_err = err
            self.best_params = params
        print("{}\t|\t{}\t|\t{}\t|\t{}\t|\t{}".format(it, self.err._value, err, params, gradient))


class MetricLearner:
    def __init__(self, tet, loss, metric, X, y):
        self.tet = tet
        self.X = X
        self.y = y
        self.metric = metric
        self.loss = loss
        self.best_params = []
        self.best_v_err = float("inf")
        self.err = -1

    def learn(self, **kwargs): 

        params = self.tet.get_params()
        optimizer = kwargs["optimizer"]

        objective_grad = grad(self.calculate_loss, argnum=0)

        self.X_train, self.X_val = self.create_triplets([3,5,7])

        print("DATASET SIZE - \t TRAIN: {} ex \t VALIDATION: {} ex".format(len(self.X_train), len(self.X_val)))

        print("Itr\t|\tTr Error\t|\tVal Error\t|\tParams\t|\tGradient\t")
        if optimizer == "adam":
            num_iters = kwargs["num_iters"]
            step_size = kwargs["step_size"]
            optimized_params = adam(objective_grad, params, step_size=step_size, num_iters=num_iters, callback=self.print_perf)
            print("\nBEST VALIDATION ERROR: ", self.best_v_err)
            print("BEST PARAMS: ", self.best_params)
            return optimized_params


    def calculate_loss(self, params, iteration):
        evaluations = []
        for triplet in self.X_train[:30]:
            #print("TARGET: ", triplet[0][0], triplet[0][1])
            #print("CLOSE: ", triplet[1][0], triplet[1][1])
            #print("FAR: ", triplet[2][0], triplet[2][1])
            close_emd = self.metric.emd(params, self.tet, triplet[0][0], triplet[1][0])
            far_emd = self.metric.emd(params, self.tet, triplet[0][0], triplet[2][0])
            evaluations.append((close_emd, far_emd))
        err = self.loss.loss(evaluations)
        self.err = err
        return err

    def print_perf(self, params, it, gradient):
        evaluations = []
        for triplet in self.X_val[:10]:
            close_emd = self.metric.emd(params, self.tet, triplet[0][0], triplet[1][0])
            far_emd = self.metric.emd(params, self.tet, triplet[0][0], triplet[2][0])
            evaluations.append((close_emd, far_emd))
        err = self.loss.loss(evaluations)
        if err < self.best_v_err:
            self.best_v_err = err
            self.best_params = params
        print("{}\t|\t{}\t|\t{}\t|\t{}\t|\t{}\t".format(it, self.err._value, err, params, gradient))

    def create_triplets(self, cls, limit=30):
        classes_list = self._divide_classes(cls)
        triplets_list = []
        for i in range(len(classes_list)):
            pairs = list(itertools.combinations(classes_list[i][:limit], 2))
            j = 0
            while j < len(pairs):
                idx = j % len(cls)
                if idx == i:
                    idx = (j + 1)%len(cls)
                ex_idx = random.randint(0,len(classes_list[idx])-1) 
                triplets_list.append((pairs[j][0], pairs[j][1], classes_list[idx][ex_idx]))
                j += 1
        np.random.shuffle(triplets_list)
        split = int(0.9 * len(triplets_list))
        triplets_train, triplets_val = triplets_list[:split], triplets_list[split:]
#        print(len(triplets_train), len(triplets_val))
        return triplets_train, triplets_val

    def _divide_classes(self, classes):
        classes_list = [[] for _ in classes]
        for x,y in zip(self.X, self.y):
            for i, c in enumerate(classes):
                if y == c:
                    classes_list[i].append((x,y))
                else:
                    pass
        return classes_list


