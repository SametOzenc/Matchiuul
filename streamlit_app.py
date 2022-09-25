import datetime as dt
from PIL import Image
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib
from catboost import CatBoostClassifier

books_data = pd.read_csv("Books.csv")
movies_data = pd.read_csv("Movies.csv")
lyrics_data = pd.read_csv("lyrics.csv")

# ----------------------------- Streamlit -----------------------------

st.set_page_config(
    page_title="Matchiuul",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)
image = Image.open('logo.jpg')

st.title("❤️SOUL MATE RECOMMENDATION ENGINE ")

col1, col2, __ = st.columns(3)
with col1:
    st.image(image, width=350)
with col2:
    st.title("")
    st.title("Welcome to M(atch)iuul")
    st.title("The future of dating")
    st.write("...powered by AI")


with st.sidebar:
    st.title("Welcome to M(atch)iuul")
    st.title("The future of dating")
    st.write("\t ...powered by AI")


col1, col2, col3 = st.columns(3)
with col1:
    user_name = st.text_input("Enter your name:")

with col2:
    user_age = st.number_input("Enter your age:")

col1, col2, col3 = st.columns(3)
with col1:
    st.expander("Gender")
    gender = ["Male", "Female"]
    user_gender = st.selectbox("Select your gender?", gender, key = 1)
with col2:
    st.expander("Status")
    status = ['Single', 'Available', 'Seeing someone', 'Married']
    user_status = st.selectbox("Select your status", status, key = 2)

col1, col2, col3 = st.columns(3)
with col1:
    st.expander("Body Type")
    body_type = ['Average', 'Fit', 'Athletic', 'Thin', 'Curvy', 'A little extra', 'Skinny', 'Full figured', 'Overweight',
                    'Jacked', 'Used up', 'Rather not say']
    user_body_type = st.selectbox("Select your body type", body_type, key = 3)
with col2:
    st.expander("Drinks")
    drinks = ['Not at all', 'Rarely', 'Socially', 'Often', 'Very often', 'Desperately']
    user_drink = st.selectbox("Select your drinking habit", drinks, key = 4)

col1, col2, col3 = st.columns(3)
with col1:
    user_height = st.number_input("Enter your height in cm:")
with col2:
    st.expander("Religion")
    religion = ['Agnosticism', 'Atheism', 'Christianity', 'Catholicism', 'Other',
       'Buddhism', 'Judaism', 'Islam']
    user_religion = st.selectbox("Select your religion", religion, key = 8)
    
col1, col2, col3 = st.columns(3)
with col1:
    st.expander("Smokes")
    smokes = ["Yes", "No"]
    user_smoke = st.selectbox("Select your smoking habit", smokes, key = 10)


col1, col2, col3 = st.columns(3)
with col1:
    st.write("Select your favorite 3 Books")
    with st.expander("Choose 3 Books", expanded=False):
        books = books_data["Kitap_Adı"].unique().tolist()
        books_chooser1 = st.selectbox("Book Name ", books, index=0)
        books_chooser2 = st.selectbox("Book Name", books, index=1)
        books_chooser3 = st.selectbox("Book Name ", books, index=2)
with col2:
    st.write("Select your favorite 3 Movies")
    with st.expander("Choose 3 Movies", expanded=False):
        movies = movies_data["Movie_Name"].unique().tolist()
        movies_chooser1 = st.selectbox("Movie Name ", movies, index=0)
        movies_chooser2 = st.selectbox("Movie Name ", movies, index=1)
        movies_chooser3 = st.selectbox("Movie Name ", movies, index=2)

col1, col2, col3 = st.columns(3)
with col1:
    st.write("Select your favorite 3 Songs")
    with st.expander("Choose 3 Songs", expanded=False):
        songs = lyrics_data["Song"].unique().tolist()
        songs_chooser1 = st.selectbox("Song Name ", songs, index=0)
        songs_chooser2 = st.selectbox("Song Name ", songs, index=1)
        songs_chooser3 = st.selectbox("Song Name ", songs, index=2)

# filter your match
col1, col2, col3 = st.columns(3)
with col1:
    st.write("Filter your match")
    with st.expander("Filter", expanded=False):
        age_filter =  st.slider("Age", 18, 99, (18, 99))
        height_filter = st.slider("Height", 140, 210, (140, 210))
        sex_filter = st.selectbox("Gender", gender, key = 20 )
        
        


# ---------------------------------------- Engine --------------------------------------------



pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 10)

df_supervised = pd.read_csv("5_cluster_df_final.csv")
df_supervised.drop(["Unnamed: 0"], axis=1, inplace=True)

machine_learning_model = joblib.load('best_model.pkl')

# Burası streamlit tarafından yeni kullanıcı olarak gelecek

if user_gender == "Male":
    user_gender = "m"
else:
    user_gender = "f"

new_profile = df_supervised[df_supervised.index == 1].drop(["Cluster"], axis=1)
new_profile["age"] = int(user_age)
new_profile["status"] = user_status.lower()
new_profile["sex"] = user_gender
new_profile["body_type"] = user_body_type.lower().replace(" ","_")
new_profile["drinks"] = user_drink.lower()
new_profile["height"] = int(user_height)
new_profile["New_Pets"] = "Likes_Boths"
new_profile["New_Job"] = "Active_working"
new_profile["New_Education"] = "Graduate_Degree"
new_profile["New_religion"] = user_religion.lower()
new_profile["New_sign"] = "Taurus"
new_profile["New_smokes"] = user_smoke


# Yeni kullanıcının model için hazırlanması

df_supervised_temp = df_supervised.append(new_profile, ignore_index=True)
df_supervised_temp = pd.get_dummies(df_supervised_temp)
new_profile_final = df_supervised_temp[df_supervised_temp.index == (df_supervised_temp.shape[0]-1)].drop(["Cluster"],
                                                                                                         axis=1)

# Yeni kullanıcının Cluster tahmin edilmesi

machine_learning_model.predict(new_profile_final)

cluster_no = machine_learning_model.predict(new_profile_final)[0][0]

new_profile["Cluster"] = cluster_no
print(new_profile)
# Yeni kullanıcının cluster'ına göre filtrelenmesi

df_supervised = df_supervised.append(new_profile, ignore_index=True)
filtered_df = df_supervised[df_supervised["Cluster"] == cluster_no]
print(filtered_df)
# ----------------------------- User - Filtered Dataframe -----------------------------

if sex_filter == "Male":
    sex_filter = "m"
else:
    sex_filter = "f"

age_low = age_filter[0]
age_up = age_filter[1]
sex_f = sex_filter
education_filter.replace(" " , "_")

final_df = filtered_df[(filtered_df["age"] > age_filter[0]) & (filtered_df["age"] < age_filter[1]) &
                           (filtered_df["sex"] == sex_filter) & (filtered_df["New_Education"] == education_filter.replace(" " , "_"))
                       &   (filtered_df["height"] > height_filter[0]) &
                           (filtered_df["height"] < height_filter[1]) ].head(25)


final_df.reset_index(drop=True, inplace=True)

# ----------------------------- TF-IDF -----------------------------
# Datasetlerin okutulması işlemi

count_vect = CountVectorizer()
vectorizer = TfidfVectorizer()


# Kitap, film, müzik gibi değişkenlerin veri setine geçici olarak sadece statü olarak eklenmesi işlemi

final_df["Books"] = final_df["status"]
final_df["Movies"] = final_df["status"]
final_df["Lyrics"] = final_df["status"]

# Her bir değişkene rastgele olarak 3 ayrı kitap, film ve müziğin okulan data setlerine eklenmesi işlemi

try:
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
except:
    st.error("We couldn't find enough users", icon="🚨")

# İndex sıralamasının bozulmasından ötürü yeniden sıralama yapılarak atılması işlemi

final_df = final_df.reset_index()

# Örnek Kitap değişkenlerinin atanması işlemi

book1 = books_chooser1
book2 = books_chooser2
book3 = books_chooser3

# Kitap özetlerinin kitap adları değişkeninden seçilip özetlerinin değişkenlere kaydedilme işlemi

book_Summary1 = books_data[books_data['Kitap_Adı'] == book1]["Summary"]
book_Summary2 = books_data[books_data['Kitap_Adı'] == book2]["Summary"]
book_Summary3 = books_data[books_data['Kitap_Adı'] == book3]["Summary"]

# 3 kitabın bir değişken içinden girilmesi işlemi

book_total_summary = (book_Summary1.values[0]+book_Summary2.values[0]+book_Summary3.values[0])

# datasetine kitap benzerliği değişkenin eklenmesi işlemi

final_df["Book_Similarity"] = final_df["status"]

# Kitap değişkenin str olarak kaydedilmesi işlemi

final_df["Books"] = final_df["Books"].astype("str")

# Kitap benzerliklerinin dataset içinde bulunan karekterler ile seçilen kitaplar arasında cosine similarity bulunması işlemi

for i in range(0,(max((final_df.index)+1))):
    books = [final_df["Books"].loc[i],book_total_summary]
    X_train_counts = count_vect.fit_transform(books)
    pd.DataFrame(X_train_counts.toarray(), columns=count_vect.get_feature_names(), index=['The_Other_Book', 'Choosing_Book'])
    trsfm = vectorizer.fit_transform(books)
    pd.DataFrame(trsfm.toarray(), columns=vectorizer.get_feature_names(), index=['The_Other_Book', 'Choosing_Book'])
    final_df["Book_Similarity"][i] = (cosine_similarity(trsfm[0:1], trsfm).tolist()[0][1])*1000

# Film değişkenlerinin örnek olarak atama işlemi

movies1 = movies_chooser1
movies2 = movies_chooser2
movies3 = movies_chooser3

# Seçilen filmlerin ayrı ayrı bir değişkene özetlerinin ekleme işlemi

movie_Summary1 = movies_data[movies_data['Movie_Name'] == movies1]["Summaries"]
movie_Summary2 = movies_data[movies_data['Movie_Name'] == movies2]["Summaries"]
movie_Summary3 = movies_data[movies_data['Movie_Name'] == movies3]["Summaries"]

# 3 farklı film özetlerinin tek bir değişken adı altında toplanması işlemi

movie_total_summary = (movie_Summary1.values[0]+movie_Summary2.values[0]+movie_Summary3.values[0])

# Film benzerlik değişkeninin veri setine eklenme işlemi

final_df["Movie_Similarity"] = final_df["status"]

# Seçilen 3 filmin veri setindeki her bir kişi için ayrı ayrı cosine similarity'sinin bulunması işlemi

for i in range(0,(max((final_df.index)+1))):
    movies = [final_df["Movies"].loc[i],movie_total_summary]
    X_train_counts = count_vect.fit_transform(movies)
    pd.DataFrame(X_train_counts.toarray(), columns=count_vect.get_feature_names(), index=['The_Other_Movie','Choosing_Movie'])
    trsfm = vectorizer.fit_transform(movies)
    pd.DataFrame(trsfm.toarray(), columns=vectorizer.get_feature_names(), index=['The_Other_Movie','Choosing_Movie'])
    final_df["Movie_Similarity"][i] = (cosine_similarity(trsfm[0:1], trsfm).tolist()[0][1])*1000

# Her bir şarkının örnek olarak atama işlemi

lyrics1 = songs_chooser1
lyrics2 = songs_chooser2
lyrics3 = songs_chooser3

# Seçilen her bir şarkının için şarkı sözlerinin bulunup bir değişkene eklenme işlemi

lyrics_Summary1 = lyrics_data[lyrics_data['Song'] == lyrics1]["Lyrics"]
lyrics_Summary2 = lyrics_data[lyrics_data['Song'] == lyrics2]["Lyrics"]
lyrics_Summary3 = lyrics_data[lyrics_data['Song'] == lyrics3]["Lyrics"]

# Her bir şarkı sözünün bir değişkene birleştirme işlemi

lyrics_total_summary = (lyrics_Summary1.values[0]+lyrics_Summary2.values[0]+lyrics_Summary3.values[0])

# Şarkı sözlerinin benzerliklerini datasete eklenmesi işlemi

final_df["Lyrics_Similarity"] = final_df["status"]

# Seçilen her bir şarkının sözü dataseti içindeki her bir kişinin seçtiği şarkı sözü ile cosine similarity hesaplama işlemi

for i in range(0,(max((final_df.index)+1))):
    lyrics = [final_df["Lyrics"].loc[i],lyrics_total_summary]
    X_train_counts = count_vect.fit_transform(lyrics)
    pd.DataFrame(X_train_counts.toarray(), columns=count_vect.get_feature_names(), index=['The_Other_Movie', 'Choosing_Movie'])
    trsfm = vectorizer.fit_transform(lyrics)
    pd.DataFrame(trsfm.toarray(), columns=vectorizer.get_feature_names(), index=['The_Other_Movie', 'Choosing_Movie'])
    final_df["Lyrics_Similarity"][i] = (cosine_similarity(trsfm[0:1], trsfm).tolist()[0][1])*1000

# Scoring process

# Toplam benzerlik değişkenin datasetine eklenme işlemi

final_df["Total_Similarity"] = final_df["Book_Similarity"]

# Her bir benzerliğin tek bir benzerlik ile belirli skorlar ile toplanıp toplam benzerlik değişkinin hesaplanma işlemi

for i in range(0,(max((final_df.index)+1))):
    final_df["Total_Similarity"][i] =  0.4*float(final_df["Book_Similarity"][i]) + 0.3 * float(final_df["Movie_Similarity"][i]) + 0.3 * float(final_df["Lyrics_Similarity"][i])


# Sorting the best 5 choices for tf-idf according to books, movies, and songs

# En iyi benzerlik oranına sahip 5 kişinin bulunması işlemi

first_five_index = final_df["Total_Similarity"].sort_values(ascending=False).index.values[5]

# Veri setinin bu 5 kişi ile filtrelenmesi işlemi

tf_idf_final_df = final_df[(final_df["Total_Similarity"] > final_df["Total_Similarity"][first_five_index])]

# Veri seti indexinin tekrardan yenilenmesi işlemi

tf_idf_final_df = tf_idf_final_df.reset_index()

# Sonradan eklenen değişkenlerin silinmesi işlemi

tf_idf_final_df.drop(["Books","Movies","Lyrics","Book_Similarity","Movie_Similarity","Lyrics_Similarity","Total_Similarity","level_0","index","Cluster"], axis=1, inplace=True)


st.dataframe(tf_idf_final_df)
st.write(tf_idf_final_df.shape)
