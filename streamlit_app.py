import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes

# Plots
import seaborn as sns
from sklearn.cluster import KMeans
from tabulate import tabulate
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
import joblib
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
from sklearn.metrics.pairwise import cosine_similarity
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)



df_supervised = pd.read_csv("Final_projesi/5_cluster_df_final.csv")
df_supervised.drop(["Unnamed: 0"], axis=1, inplace=True)
df_supervised.head()
df_supervised.New_Education.unique()

machine_learning_model = joblib.load('best_model.pkl')

# Burası streamlit tarafından yeni kullanıcı olarak gelecek

new_profile = df_supervised[df_supervised.index == 1].drop(["Cluster"], axis=1)
new_profile["age"] = 20
new_profile["sex"] = "f"
new_profile["height"] = 165
new_profile["New_smokes"] = "Yes"
pd.get_dummies(new_profile, drop_first=False)

# Yeni kullanıcının model için hazırlanması

df_supervised_temp = df_supervised.append(new_profile, ignore_index=True)
df_supervised_temp = pd.get_dummies(df_supervised_temp)
new_profile_final = df_supervised_temp[df_supervised_temp.index == (df_supervised_temp.shape[0]-1)].drop(["Cluster"],
                                                                                                         axis=1)

# Yeni kullanıcının Cluster tahmin edilmesi

machine_learning_model.predict(new_profile_final)

cluster_no = joblib.load('best_model.pkl').predict(new_profile_final)[0][0]
new_profile["Cluster"] = cluster_no

# Yeni kullanıcının cluster'ına göre filtrelenmesi

df_supervised = df_supervised.append(new_profile, ignore_index=True)
filtered_df = df_supervised[df_supervised["Cluster"] == cluster_no]

# ----------------------------- User - Filtered Dataframe -----------------------------

age_low = 18
age_up = 25
sex = "m"
New_religion = "atheism"

final_df = filtered_df[(filtered_df["age"] > age_low) & (filtered_df["age"] < age_up) &
                           (filtered_df["sex"] == sex) & (filtered_df["New_religion"] == New_religion)]


final_df.reset_index(drop=True, inplace=True)

# ----------------------------- TF-IDF -----------------------------
# Datasets

books_data = pd.read_csv("Final_projesi/Tf-idf_ready/Books.csv")
movies_data = pd.read_csv("Final_projesi/Tf-idf_ready/Movies.csv")
lyrics_data = pd.read_csv("Final_projesi/Tf-idf_ready/lyrics.csv")

# Adding of variables: Books, Movies, and Lyrics

final_df["Books"] = final_df["status"]
final_df["Movies"] = final_df["status"]
final_df["Lyrics"] = final_df["status"]

# Assign a random values for this variables


for i in range(0, (max(final_df.index + 1))):
    book_total_summary = [books_data.sample(1)["Summary"].values[0] + books_data.sample(1)["Summary"].values[0] +
                          books_data.sample(1)["Summary"].values[0]]
    final_df["Books"][i] = book_total_summary

    movie_frames = (movies_data.sample(1)["Summaries"].values[0] + movies_data.sample(1)["Summaries"].values[0] +
                    movies_data.sample(1)["Summaries"].values[0])
    final_df["Movies"][i] = movie_frames

    lyrics_frames = (lyrics_data.sample(1)["Lyrics"].values[0] + lyrics_data.sample(1)["Lyrics"].values[0] +
                     lyrics_data.sample(1)["Lyrics"].values[0])
    final_df["Lyrics"][i] = lyrics_frames

# Refresh of index in the dataframe

final_df = final_df.reset_index()

# Calculate the book similarity

book1 = "Savaş ve Barış"
book2 = "Hamlet"
book3 = "Harry Potter ve Felsefe Taşı"

book_Summary1 = books_data[books_data['Kitap_Adı'] == book1]["Summary"]
book_Summary2 = books_data[books_data['Kitap_Adı'] == book2]["Summary"]
book_Summary3 = books_data[books_data['Kitap_Adı'] == book3]["Summary"]


book_total_summary = (book_Summary1.values[0]+book_Summary2.values[0]+book_Summary3.values[0])

final_df["Book_Similarity"] = final_df["status"]

for i in range(0,(max((final_df.index)+1))):
    books = [final_df["Books"].loc[i],book_total_summary]
    X_train_counts = count_vect.fit_transform(books)
    pd.DataFrame(X_train_counts.toarray(), columns=count_vect.get_feature_names(), index=['Choosing_Book', 'The_Other_Book'])
    trsfm = vectorizer.fit_transform(books)
    pd.DataFrame(trsfm.toarray(), columns=vectorizer.get_feature_names(), index=['Choosing_Book', 'The_Other_Book'])
    final_df["Book_Similarity"][i] = (cosine_similarity(trsfm[0:1], trsfm).tolist()[0][1])*1000

movies1 = "Avatar"
movies2 = "Titanic"
movies3 = "Transformers: Dark of the Moon"

movie_Summary1 = movies_data[movies_data['Movie_Name'] == movies1]["Summaries"]
movie_Summary2 = movies_data[movies_data['Movie_Name'] == movies2]["Summaries"]
movie_Summary3 = movies_data[movies_data['Movie_Name'] == movies3]["Summaries"]

movie_total_summary = (movie_Summary1.values[0]+movie_Summary2.values[0]+movie_Summary3.values[0])

final_df["Movie_Similarity"] = final_df["status"]

for i in range(0,(max((final_df.index)+1))):
    movies = [final_df["Movies"].loc[i],movie_total_summary]
    X_train_counts = count_vect.fit_transform(movies)
    pd.DataFrame(X_train_counts.toarray(), columns=count_vect.get_feature_names(), index=['Choosing_Movie', 'The_Other_Movie'])
    trsfm = vectorizer.fit_transform(movies)
    pd.DataFrame(trsfm.toarray(), columns=vectorizer.get_feature_names(), index=['Choosing_Movie', 'The_Other_Movie'])
    final_df["Movie_Similarity"][i] = (cosine_similarity(trsfm[0:1], trsfm).tolist()[0][1])*1000


lyrics1 = "Revolution"
lyrics2 = "Hey Okay"
lyrics3 = "The devil you know"

lyrics_Summary1 = lyrics_data[lyrics_data['Song'] == lyrics1]["Lyrics"]
lyrics_Summary2 = lyrics_data[lyrics_data['Song'] == lyrics2]["Lyrics"]
lyrics_Summary3 = lyrics_data[lyrics_data['Song'] == lyrics3]["Lyrics"]

lyrics_total_summary = (lyrics_Summary1.values[0]+lyrics_Summary2.values[0]+lyrics_Summary3.values[0])

final_df["Lyrics_Similarity"] = final_df["status"]

for i in range(0,(max((final_df.index)+1))):
    lyrics = [final_df["Lyrics"].loc[i],lyrics_total_summary]
    X_train_counts = count_vect.fit_transform(lyrics)
    pd.DataFrame(X_train_counts.toarray(), columns=count_vect.get_feature_names(), index=['Choosing_Movie', 'The_Other_Movie'])
    trsfm = vectorizer.fit_transform(lyrics)
    pd.DataFrame(trsfm.toarray(), columns=vectorizer.get_feature_names(), index=['Choosing_Movie', 'The_Other_Movie'])
    final_df["Lyrics_Similarity"][i] = (cosine_similarity(trsfm[0:1], trsfm).tolist()[0][1])*1000

# Scoring process

final_df["Total_Similarity"] = final_df["Book_Similarity"]

for i in range(0,(max((final_df.index)+1))):
    final_df["Total_Similarity"][i] =  0.4*float(final_df["Book_Similarity"][i]) + 0.3 * float(final_df["Movie_Similarity"][i]) + 0.3 * float(final_df["Lyrics_Similarity"][i])


# Sorting the best 5 choices for tf-idf according to books, movies, and songs

first_index = final_df["Total_Similarity"].sort_values(ascending=False).index.values[6]

tf_idf_final_df = final_df[(final_df["Total_Similarity"] > final_df["Total_Similarity"][first_index])]

tf_idf_final_df = tf_idf_final_df.reset_index()

# tf_idf_final_df.drop(["Books","Movies","Lyrics","Book_Similarity","Movie_Similarity","Lyrics_Similarity","Total_Similarity","level_0","index","Cluster"], axis=1, inplace=True)

tf_idf_final_df
