import pandas as pd
import matrix_factorization as MF

def dataset_1():
    data_frame = pd.read_csv('user_artists_hetrec.dat', delimiter='\t')

    #Put data into matrix
    users = data_frame['userID'].values
    items = data_frame['artistID'].values
    weights = data_frame['weight'].values
    
    model = MF.MF(users, items, weights)
    result = [] #List of calculated RMSE between model prediction and test set for each number of latent factors

    for i in range(5,101):
        model.train(n_latent_factors=i)
        result.append([i, model.evaluate])

    return result

def dataset_2():
    data_frame

def main():
    hetrec_result = dataset_1()
    
    print("Done")
if __name__ == "__main__":
    main()