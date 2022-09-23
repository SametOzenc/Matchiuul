import datetime as dt
from PIL import Image
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib
from catboost import CatBoostClassifier



# ----------------------------- Streamlit -----------------------------

st.set_page_config(
    page_title="Matchiuul",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
image = Image.open(r'C:\Users\Sam\PycharmProjects\VBO\Final_projesi\logo.jpg')

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
    user_gender = st.selectbox("Select your gender?", gender)
with col2:
    st.expander("Status")
    status = ['Single', 'Available', 'Seeing someone', 'Married']
    user_status = st.selectbox("Select your status", status)

col1, col2, col3 = st.columns(3)
with col1:
    st.expander("Body Type")
    body_type = ['Average', 'Fit', 'Athletic', 'Thin', 'Curvy', 'A little extra', 'Skinny', 'Full figured', 'Overweight',
                    'Jacked', 'Used up', 'Rather not say']
    user_body_type = st.selectbox("Select your body type", body_type)
with col2:
    st.expander("Drinks")
    drinks = ['Not at all', 'Rarely', 'Socially', 'Often', 'Very often', 'Desperately']
    user_drink = st.selectbox("Select your drinking habit", drinks)

col1, col2, col3 = st.columns(3)
with col1:
    st.expander("Pets")
    education = ['Likes Boths', 'Only Cat', 'Only Dog', "Don't Like it"]
    user_pets = st.selectbox("Select your pets preference", education)
with col2:
    user_height = st.number_input("Enter your height in cm:")

col1, col2, col3 = st.columns(3)
with col1:
    st.expander("Job")
    job = ["Student", "Active working", "Retired"]
    user_job = st.selectbox("Select your job status", job)
with col2:
    st.expander("Education")
    education = ["Postgraduate degree", "Graduate Degree", "High School", "Dropped out of High School"]
    user_education = st.selectbox("Select your education level", education)

col1, col2, col3 = st.columns(3)
with col1:
    st.expander("Religion")
    religion = ['Agnosticism', 'Atheism', 'Christianity', 'Catholicism', 'Other',
       'Buddhism', 'Judaism', 'Islam']
    user_religion = st.selectbox("Select your religion", religion)
with col2:
    st.expander("New_sign")
    new_sign = ['Gemini', 'Cancer', 'Taurus', 'Sagittarius', 'Leo', 'Aquarius',
       'Libra', 'Pisces', 'Scorpio', 'Aries', 'Capricorn', 'Virgo']
    user_sign = st.selectbox("Select your new_sign", new_sign)

col1, col2, col3 = st.columns(3)
with col1:
    st.expander("Smokes")
    smokes = ["Yes", "No"]
    user_smoke = st.selectbox("Select your smoking habit", smokes)

col1, col2, col3 = st.columns(3)
with col1:
    st.write("Select your favorite 3 Books")
    with st.expander("Choose 3 Books", expanded=False):
        books = ["Harry Potter", "Lord of the Rings", "Dune"]
        chooser1 = st.selectbox("Book Name ", books, index=0, key=1)
        chooser2 = st.selectbox("Book Name", books, index=1, key=2)
        chooser3 = st.selectbox("Book Name ", books, index=2, key=3)
with col2:
    st.write("Select your favorite 3 Movies")
    with st.expander("Choose 3 Movies", expanded=False):
        books = ["Esaretin Bedeli", "Matrix", "Batman"]
        chooser1 = st.selectbox("Movie Name ", books, index=0, key=1)
        chooser2 = st.selectbox("Movie Name ", books, index=1, key=2)
        chooser3 = st.selectbox("Movie Name ", books, index=2, key=3)

col1, col2, col3 = st.columns(3)
with col1:
    st.write("Select your favorite 3 Music")
    with st.expander("Choose 3 Songs", expanded=False):
        books = ["Kuzu Kuzu", "Dudu", "Öp"]
        chooser1 = st.selectbox("Song Name ", books, index=0, key=1)
        chooser2 = st.selectbox("Song Name ", books, index=1, key=2)
        chooser3 = st.selectbox("Song Name ", books, index=2, key=3)

# filter your match
col1, col2, col3 = st.columns(3)
with col1:
    st.write("Filter your match")
    with st.expander("Filter", expanded=False):
        age_filter =  st.slider("Age", 18, 99, (18, 99))
        height_filter = st.slider("Height", 140, 210, (140, 210))
        sex_filter = st.selectbox("Gender", gender )
        education_filter = st.selectbox("Education", education)
        religion_filter = st.selectbox("Religion", religion)


# ---------------------------------------- Engine --------------------------------------------



pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 10)

df_supervised = pd.read_csv(r"C:\Users\Sam\PycharmProjects\VBO\Final_projesi\Tf-idf_ready\5_cluster_df_final.csv")
df_supervised.drop(["Unnamed: 0"], axis=1, inplace=True)

machine_learning_model = joblib.load(r'C:\Users\Sam\PycharmProjects\VBO\Final_projesi\Tf-idf_ready\best_model.pkl')

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
new_profile["New_Pets"] = user_pets.replace(" " , "_")
new_profile["New_Job"] = user_job.replace(" " , "_")
new_profile["New_Education"] = user_education.replace(" " , "_")
new_profile["New_religion"] = user_religion.lower()
new_profile["New_sign"] = user_sign
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
                       & (filtered_df["New_religion"] == religion_filter.lower()) & (filtered_df["height"] > height_filter[0]) &
                           (filtered_df["height"] < height_filter[1]) ]

st.dataframe(final_df)
st.write(final_df.shape)

%conda env export > environment.yaml