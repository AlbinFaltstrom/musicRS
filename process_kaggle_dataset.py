import pandas as pd
import csv

userid_table = dict()
trackid_table = dict()
user_id_count = 0
track_id_count = 0

df = pd.read_csv('user_listening_history_kaggle.csv')
df = df.sample(frac=1/10).reset_index(drop=True)

with open('kaggle_numbered_10%.csv', 'w', newline='') as file1:
    wr = csv.writer(file1)
    wr.writerow(['track_id', 'user_id','playcount'])

    for index, row in df.iterrows():
        t_id = 0
        u_id = 0

        if not trackid_table.__contains__(row['track_id']):
            track_id_count += 1
            trackid_table[row['track_id']] = track_id_count

        if not userid_table.__contains__(row['user_id']):
            user_id_count += 1
            userid_table[row['user_id']] = user_id_count

        t_id = trackid_table[row['track_id']]
        u_id = userid_table[row['user_id']]

        wr.writerow([t_id, u_id, row['playcount']])