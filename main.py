
#Read data
data_frame = pd.read_csv('user_artists_hetrec.dat', delimiter='\t')

#Put data into matrix
users = data_frame['userID'].values
items = data_frame['artistID'].values
weights = data_frame['weight'].values

def main():
    print("Done")
if __name__ == "__main__":
    main()