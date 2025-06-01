import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def load_data(input_file):
    df = pd.read_csv(input_file)

    if 'title' not in df.columns:
        raise ValueError("Kolom 'title' tidak ditemukan dalam dataset!")

    df['title'] = df['title'].fillna('')
    return df

def transform_tfidf(titles):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(titles)
    return tfidf_matrix, vectorizer

def save_tfidf_matrix(tfidf_matrix, vectorizer, output_file):
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    df_tfidf.to_csv(output_file, index=False)

def save_vectorizer(vectorizer, model_file):
    joblib.dump(vectorizer, model_file)

if __name__ == "__main__":
    # Path file input/output
    input_file = os.path.join('..', 'data', 'preprocessed.csv')
    output_file = os.path.join('..', 'data', 'tfidf.csv')
    tfidf_model_file = os.path.join('..', 'model', 'tfidf_vectorizer.pkl')

    # Load data
    print("Memuat data preprocessed...")
    df = load_data(input_file)

    # Transformasi TF-IDF
    print("Melakukan transformasi TF-IDF...")
    tfidf_matrix, tfidf_vectorizer = transform_tfidf(df['title'])

    # Simpan hasil TF-IDF dan model
    save_tfidf_matrix(tfidf_matrix, tfidf_vectorizer, output_file)
    print(f"Hasil TF-IDF disimpan di {output_file}")

    save_vectorizer(tfidf_vectorizer, tfidf_model_file)
    print(f"Model TF-IDF disimpan di {tfidf_model_file}")
