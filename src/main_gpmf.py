import numpy as np
import sys
import pandas
from gpmf import factorize

args=list(sys.argv)
test_input =args[2]
train_input =args[1]
steps = int(args[3])

train = pandas.read_csv(train_input).sort_values(by=['user_id','movie_id'])
test = pandas.read_csv(test_input).sort_values(by=['user_id','movie_id'])
users = train['user_id'].astype(np.int32)
movies = train['movie_id'].astype(np.int32)
ratings = train['rating']
test_users = test['user_id'].astype(np.int32)
test_movies = test['movie_id'].astype(np.int32)
test_ratings = test['rating']

factorize(users, movies, ratings, test_users, test_movies, test_ratings, latent=30, steps=steps, debug=2)
