import csv
import pandas as pd
import matrix_factorization as MF

def run_experiment_dataset1(min_lf, max_lf):
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

    for i in range(min_lf, max_lf + 1):
        model.train(n_latent_factors=i)
        result.append(model.evaluate())
        print("Models trained: " + str(i))

    return result

def run_experiment_dataset2(min_lf, max_lf):
    data_frame = pd.read_csv('kaggle_numbered_10%.csv')

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

    for i in range(min_lf, max_lf + 1):
        model.train(n_latent_factors=i)
        result.append(model.evaluate())
        print("Models trained: " + str(i))

    return result

def save_result(result, filename):
    with open(filename, 'w') as file:
        wr = csv.writer(file)
        wr.writerow(result)

def main():
    save_result(run_experiment_dataset1(5,50), 'hetrec_result.txt')
    save_result(run_experiment_dataset2(5,50), 'kaggle_result.txt')

if __name__ == "__main__":
    main()