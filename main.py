import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import root_mean_squared_error

#Read data
data_frame = pd.read_csv('user_artists_hetrec.dat', delimiter='\t')

#Put data into matrix
users = data_frame['userID'].values
artists = data_frame['artistID'].values
weights = data_frame['weight'].values
#obs = [(users[i]-1, artists[i]-1) for i in range(len(users))]
matrix = sparse.coo_matrix((weights, (users, artists))).tocsr()

#Parameters
n_latent_factors = 5
max_iter = 100
step_size = 0.1
regularization = 0.01

#Training and test sets (TBD)
train = "" #training set
test = "" #test set

#Initialize user and item vectors to random values
user_vectors = np.random.normal(size=(np.max(users-1), n_latent_factors))
item_vectors = np.random.normal(size=(np.max(artists), n_latent_factors))
print(user_vectors.shape)
print(user_vectors[:5])

#Loss function
# u = user, i = item
def calc_error(u, i):
    return root_mean_squared_error(vMatrix(u,i))

#SGD
def SGD():
    for i in range(maxiter):
        for l in range():
            i, j = l
            error = ( np.dot(U[i, :], V[j, :].T) - M[i, j] )
            U_temp = np.copy(U[i, :])
            # update U
            U[i, :] -= alpha * (error*V[j, :] + beta*U[i, :])
            # update V
            V[j, :] -= alpha * (error*U_temp + beta*V[j, :])
        errorVec[0, i] = MSE(U, V, M, test)
        errorVec[1, i] = RelError(U, V, M, test)





def main():
    print("Done")
if __name__ == 'main':
    main()