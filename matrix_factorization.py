import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import root_mean_squared_error

class MF():

    def __init__(self, users, items, weights):

        #Put data into matrix
        self.matrix = sparse.coo_matrix((weights, (users, items))).tocsr()

        #Training and test sets (TODO add seed when dividing into sets)
        self.set = [(users[i], items[i]) for i in range(len(users))]
        np.random.shuffle(self.set)
        self.training = self.set[0:(int)(np.floor(0.75*len(self.set)))] # training set, 75% of data
        self.test = self.set[(int)(np.floor(0.75*len(self.set))):len(self.set)+1] # test set, 25% of data

        #Parameters
        self.max_passes = 100
        self.step_size = 0.1
        self.reg = 0.01 #Regularization factor

    def train(self, n_latent_factors):
        self.n_latent_factors = n_latent_factors
        #Initialize user and item vectors to random values
        self.user_vectors = np.random.normal(size=(np.max(self.users), self.n_latent_factors))
        self.item_vectors = np.random.normal(size=(np.max(self.items), self.n_latent_factors))

        for i in range(self.max_passes):
            #Shuffle the training set
            np.random.shuffle(self.training)
            self.SGD()

    def SGD(self):
        for i in self.training:
            u, m = self.training[i]
            error = self.calc_error(u, m)
            temp = self.user_vectors
            self.user_vectors[u, :] -= self.step_size * (error * self.item_vectors - self.reg * self.user_vectors[u, :])
            self.item_vectors[m, :] -= self.step_size * (error * temp - self.reg * self.item_vectors[m, :])

    def calc_error(self, u, m):
        prediction = np.dot(self.users(u), self.items(m))
        actual = self.matrix[u, m]
        return actual - prediction
    
    def get_rating(self, user, item):
        return np.dot(self.user_vectors[user], self.item_vectors[item])

    def evaluate(self):
        predictions = []
        actual = []
        for i in self.test:
            u, m = self.test[i]
            predictions.append(self.get_rating(u, m))
            actual.append(self.matrix[u, m])
        return root_mean_squared_error(actual, predictions)