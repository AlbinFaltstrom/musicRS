import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import root_mean_squared_error

class matrix_factorization():

    def __init__(self, users, items, weights, n_latent_factors):

        #Put data into matrix
        self.matrix = sparse.coo_matrix((weights, (users, items))).tocsr()

        #Training and test sets (TODO add seed when dividing into sets)
        self.set = [(users[i], items[i]) for i in range(len(users))]
        np.random.shuffle(set)
        self.train = set[0:(int)(np.floor(0.75*len(set)))] # training set, 75% of data
        self.test = set[(int)(np.floor(0.75*len(set))):len(set)+1] # test set, 25% of data

        #Parameters
        self.n_latent_factors = n_latent_factors
        self.max_passes = 100
        self.step_size = 0.1
        self.reg = 0.01 #Regularization factor

    def train(self):

        #Initialize user and item vectors to random values
        self.user_vectors = np.random.normal(size=(np.max(self.users), self.n_latent_factors))
        self.item_vectors = np.random.normal(size=(np.max(self.items), self.n_latent_factors))

        for i in range(self.max_passes):
            #Shuffle the training set
            np.random.shuffle(self.train)
            self.SGD()

    def SGD(self):
        for i in self.train:
            u, m = self.train[i]
            error = self.calc_error(u, m)
            temp = self.user_vectors
            self.user_vectors[u, :] -= self.step_size * (error * self.item_vectors - self.reg * self.user_vectors[u, :])
            self.item_vectors[m, :] -= self.step_size * (error * temp - self.reg * self.item_vectors[m, :])

    def calc_error(self, u, m):
        prediction = np.dot(self.users(u), self.items(m))
        actual = self.matrix[u, m]
        return actual - prediction