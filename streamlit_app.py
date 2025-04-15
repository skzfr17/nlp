import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Load model dan data
loaded_model = joblib.load('model_svc.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')
df = pd.read_csv('dataset_cleaned.csv',on_bad_lines="skip", sep=";")
tfidf_matrix = loaded_vectorizer.transform(df['title'])



# Fungsi prediksi
def prediksi_kategori(artikel_baru):
    artikel_vector = loaded_vectorizer.transform([artikel_baru])
    pred_numeric = loaded_model.predict(artikel_vector)
    return le.inverse_transform([int(pred_numeric[0])])[0]

# Fungsi rekomendasi
def rekomendasi_artikel(artikel_baru, top_n=3):
    kategori_prediksi = prediksi_kategori(artikel_baru)
    artikel_vector = loaded_vectorizer.transform([artikel_baru])
    cosine_similarities = cosine_similarity(artikel_vector, tfidf_matrix).flatten()
    
    kategori_sama_indices = df[df['kategori'] == kategori_prediksi].index.tolist()
    if not kategori_sama_indices:
        return kategori_prediksi, pd.DataFrame()

    kategori_sama_tfidf_indices = [df.index.get_loc(i) for i in kategori_sama_indices]
    kategori_sama_similarities = [(i, cosine_similarities[i]) for i in kategori_sama_tfidf_indices]
    kategori_sama_similarities.sort(key=lambda x: x[1], reverse=True)

    similar_indices = [i[0] for i in kategori_sama_similarities[:top_n]]

    rekomendasi_df = pd.DataFrame([
        {
            "Judul Artikel": df.iloc[i]['title'],
            "Kategori": df.iloc[i]['kategori'],
            "Cuplikan Konten": df.iloc[i]['content'][:150] + "...",
            "Cosine Similarity": round(cosine_similarities[i], 3)
        }
        for i in similar_indices
    ])

    return kategori_prediksi, rekomendasi_df


#Streamlit UI
st.title("Sistem Rekomendasi Artikel Berita")
st.write("Masukkan teks artikel baru, dan sistem akan memprediksi kategorinya serta merekomendasikan artikel terkait.")

input_artikel = st.text_area("Input Artikel Baru", height=200)

if st.button("Prediksi & Rekomendasi"):
    if input_artikel.strip() != "":
        kategori, rekomendasi = rekomendasi_artikel(input_artikel)
        st.success(f"Kategori Prediksi: {kategori}")
        st.write("Rekomendasi Artikel Serupa:")
        st.dataframe(rekomendasi)
    else:
        st.warning("Tolong isi teks artikel terlebih dahulu.")