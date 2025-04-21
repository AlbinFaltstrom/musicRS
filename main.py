import pandas as pd
import matrix_factorization as MF

def dataset_1():
    data_frame = pd.read_csv('user_artists_hetrec.dat', delimiter='\t')

    #Put data into matrix
    users = data_frame['userID'].values
    items = data_frame['artistID'].values
    weights = data_frame['weight'].values

    #Rescale data entries to a value between 0 and 1 (min-max normalization)
    w_min = min(weights)
    w_max = max(weights)
    weights = [(x-w_min)/w_max-w_min for x in weights]
    
    model = MF.MF(users, items, weights)
    result = [] #List of calculated RMSE between model prediction and test set for each number of latent factors

    for i in range(6,7):
        model.train(n_latent_factors=i)
        result.append([i, model.evaluate()])

    return result

def dataset_2():
    data_frame = pd.read_csv('user_listening_history_kaggle')

    #Put data into matrix
    users = data_frame['user_id'].values
    items = data_frame['track_id'].values
    weights = data_frame['playcount'].values
    
    #Rescale data entries to a value between 0 and 1 (min-max normalization)
    w_min = min(weights)
    w_max = max(weights)
    weights = [(x-w_min)/w_max-w_min for x in weights]

    model = MF.MF(users, items, weights)
    result = [] #List of calculated RMSE between model prediction and test set for each number of latent factors

    for i in range(5,101):
        model.train(n_latent_factors=i)
        result.append([i, model.evaluate])

    return result

def main():
    hetrec_result = dataset_1()
    print(hetrec_result)
    print("Dataset 1 done")
    #kaggle_result = dataset_2()
    print("All done")
if __name__ == "__main__":
    main()