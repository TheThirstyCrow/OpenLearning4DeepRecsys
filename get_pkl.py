import time
import numpy as np
from six import next
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import scipy
import pickle
#import _pickle as cPickle
import codecs

def get_100k_data():
	df = pd.read_csv(r"ratings.csv", sep=',', engine='python')

	df["rating"] = df["rating"].astype(np.float32)

	user_mapping = {}
	movie_mapping = {}
	index = 0
	for x in list(df["userId"].unique()):
		user_mapping[x] = index
		index += 1
	index = 0
	for x in list(df["movieId"].unique()):
		movie_mapping[x] = index
		index += 1

	df["userId"] = df["userId"].map(user_mapping)

	df["movieId"] = df["movieId"].map(movie_mapping)
	#for col in ("userId", "movieId"):
	#    df[col] = df[col].astype(np.int32)

	movies = pd.read_csv(r"movies.csv", sep=',', engine='python')
	movies["movieId"]= movies["movieId"].map(movie_mapping)
	movies = movies.set_index('movieId')
	movies["genres"]= movies["genres"].map(lambda x: x.replace('|', ' ').lower())
	#vectorizer = CountVectorizer(binary = True)
	#vectorizer = vectorizer.fit(list(movies["genres"]))
	#movies["genres"]= movies["genres"].map(lambda x: vectorizer.transform([x]))
	movie_content = []
	index_set = set(movies.index)
	for i in range(len(movie_mapping)):
		if i in index_set:
			movie_content.append(movies.loc[[i]].iloc[0]["genres"])
		else:
			movie_content.append('')

	vectorizer = CountVectorizer(binary = True)
	movie_content = vectorizer.fit_transform(movie_content)
	movie_content = movie_content.astype(np.float32)

	users = pd.read_csv(r"tags.csv", sep=',', engine='python')
	users["userId"]= users["userId"].map(user_mapping)
	users = users.set_index('userId')
	user_content = []
	index_set = set(users.index)
	for i in range(len(user_mapping)):
		if i in index_set:
			user_content.append(' '.join(list(users.loc[[i]]["tag"])))
		else:
			user_content.append('')
	user_content = vectorizer.fit_transform(user_content)
	user_content = user_content.astype(np.float32)

	#users = pd.DataFrame(users.groupby('userId')['tag'].agg(lambda x: ' '.join(x)).reset_index(name = "tags"))
	#vectorizer = CountVectorizer(binary = True)
	#vectorizer = vectorizer.fit(list(users["tags"]))
	#users["tags"]= users["tags"].map(lambda x: vectorizer.transform([x]))

	print("%%%%%%%%%%%%%%%%%%%%%%%%%")
	print(df["userId"])
	print("%%%%%%%%%%%%%%%%%%%%%%%%%")

	df_n ={}
	df_n["user_1"] = df["userId"][: int(len(df["userId"])/2)]
	df_n["user_2"] = df["userId"][int(len(df["userId"])/2):]
	df_n["user_1"].index = range(int(len(df_n["user_1"])))
	df_n["user_2"].index = range(int(len(df_n["user_2"])))
	print(df_n["user_1"])
	df_n["item_1"] = df["movieId"][: int(len(df["movieId"])/2)]
	df_n["item_2"] = df["movieId"][int(len(df["movieId"])/2):]
	df_n["item_1"].index = range(int(len(df_n["item_1"])))
	df_n["item_2"].index = range(int(len(df_n["item_2"])))

	df_n["rate_1"] = df["movieId"][: int(len(df["rating"])/ 2)]
	df_n["rate_2"] = df["movieId"][int(len(df["rating"])/2):]
	df_n["rate_1"].index = range(int(len(df_n["rate_1"])))
	df_n["rate_2"].index = range(int(len(df_n["rate_2"])))

	df_n = pd.DataFrame(data=df_n)
	print("Before:", df_n.shape)

	df_n = df_n.dropna(axis = 0, how="all")
	print("After:", df_n.shape)

	# l = int(len(df["userId"])
	# df_n={}
	# df_n["user_1"] = df["userId"].iloc[:int(l/2)]
	# df_n["user_2"] = df["userId"].iloc[int(l/2):]
	# print("len of user_1",int(len(df_n["user_1"]))
	# print("len of user_2", int(len(df_n["user_2"]))
	#
	# l = int(len(df["movieId"])
	# df_n["item_1"] = df["movieId"].iloc[:int(l/2)]
	# df_n["item_2"] = df["movieId"].iloc[int(l/2):]
	# print("len of movie_1", int(len(df_n["item_1"]))
	# print("len of movie_2", int(len(df_n["item_2"]))
	#
	# df_n["rate_1"] = df["rating"].iloc[:int(l/2)]
	# df_n["rate_2"] = df["rating"].iloc[int(l/ 2):]
	# print("len of rate_1", int(len(df_n["rate_1"].index))
	# print("len of rate_2", int(len(df_n["rate_2"].index))
	# print("again user_1 length: ", int(len(df_n["user_1"]))
	#
	# df_n = pd.DataFrame(data=df_n)

	# df.drop(["userId", "movieId", "rating"], axis =1)
	# df = df.rename(columns={"userId":"user", "movieId":"item", "rating":"rate"})

	rows = int(len(df_n))
	print("total rows=", rows)
	df_n = df_n.iloc[np.random.permutation(rows)].reset_index(drop=True)
	split_index_train = int(rows * 0.8)
	split_index_val = int(rows*0.9)
	df_train = df_n[0:split_index_train]
	df_val = df_n[split_index_train : split_index_val]
	df_test = df_n[split_index_val:].reset_index(drop=True)

	print(len(df_n["rate_1"]))
	print(len(df_n["rate_2"]))
	print(len(df_n["item_1"]))
	print(len(df_n["item_2"]))
	print(len(df_n["user_1"]))
	print(len(df_n["user_2"]))
	print(split_index_train)
	print(split_index_val)
	print("&&&&&&&&&&&&&&&&&&&&&&&&&")
	print(df_n["rate_1"])
	print("&&&&&&&&&&&&&&&&&&&&&&&&&")


	with codecs.open('cross_movielens_100k.pkl', 'wb') as outfile:
		pickle.dump((df_train,df_val, df_test,user_content,movie_content), outfile, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
	get_100k_data()
	print("Done!")