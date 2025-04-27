import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import root_mean_squared_error

class MF():

    def __init__(self, users, items, weights):

        #Put data into matrix
        self.matrix = sparse.coo_matrix((weights, (users, items))).tocsr()

        #Save number of users and items
        self.n_users = len(users)
        self.n_items = len(items)

        #Training and test sets (TODO add seed when dividing into sets)
        self.set = [(users[i], items[i]) for i in range(len(users))]
        np.random.shuffle(self.set)
        self.training = self.set[0:(int)(np.floor(0.75*len(self.set)))] # training set, 75% of data
        self.test = self.set[(int)(np.floor(0.75*len(self.set))):len(self.set)+1] # test set, 25% of data

        #Parameters
        self.max_passes = 100
        self.step_size = 0.1
        self.reg = 0.01 #Regularization factor
        self.batch_size = np.floor(len(self.training)*0.1) #batch size of 10%

    def train(self, n_latent_factors):
        self.n_latent_factors = n_latent_factors
        #Initialize user and item vectors to random values
        self.user_vectors = np.random.normal(size=(self.n_users, self.n_latent_factors), scale=1/n_latent_factors)
        self.item_vectors = np.random.normal(size=(self.n_items, self.n_latent_factors), scale=1/n_latent_factors)
        self.prev_RMSE = 0

        for i in range(self.max_passes):
            #Shuffle the training set
            np.random.shuffle(self.training)
            batch = self.training[0:(int)(self.batch_size)]

            self.SGD(batch)
            #Every 10th loop, check if error converges (within a margin)
            if (i%10 == 0):
                current_RMSE = self.calc_RMSE()
                if (np.abs(self.prev_RMSE - current_RMSE) < 0.001):
                    break
                else:
                    self.prev_RMSE = current_RMSE

    def SGD(self, batch):
        for i in batch:
            u, m = i
            error = self.calc_error(u, m)

            temp = np.copy(self.user_vectors[u, :])
            self.user_vectors[u, :] -= self.step_size * (error * self.item_vectors[m, :] + self.reg * self.user_vectors[u, :])
            self.item_vectors[m, :] -= self.step_size * (error * temp + self.reg * self.item_vectors[m, :])

    def calc_error(self, u, m):
        prediction = self.user_vectors[u, :].dot(self.item_vectors[m, :])
        actual = self.matrix[u, m]
        return prediction - actual

    def calc_RMSE(self):
        predictions = []
        actual = []
        for i in self.training:
            u, m = i
            predictions.append(self.get_rating(u, m))
            actual.append(self.matrix[u, m])
        return root_mean_squared_error(actual, predictions)

    def evaluate(self):
        predictions = []
        actual = []
        for i in self.test:
            u, m = i
            predictions.append(self.get_rating(u, m))
            actual.append(self.matrix[u, m])
        return root_mean_squared_error(actual, predictions)
    
    def get_rating(self, user, item):
        return np.dot(self.user_vectors[user, :], self.item_vectors[item, :])