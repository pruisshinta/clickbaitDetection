import os
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import joblib

nltk.download('stopwords')
nltk.download('punkt')

# Load kamus normalisasi
def load_alay_dictionary(file_path):
    alay_dict = pd.read_csv(file_path, encoding='ISO-8859-1')
    return dict(zip(alay_dict['slang'], alay_dict['formal']))

def case_folding(text):
    text = text.lower()
    return text

def clean_text(text):
    text = re.sub(r'#\w+', '', text)  # Menghapus hashtag
    text = re.sub(r'@\w+', '', text)  # Menghapus mention
    text = re.sub(r'[^\w\s]', ' ', text)  # Menghapus tanda baca
    text = re.sub(r'\d+', '', text)  # Menghapus angka
    text = re.sub(r'\s+', ' ', text).strip()  # Menghapus spasi berlebih
    return text

def normalize_text(text, alay_dict):
    words = text.split()
    return ' '.join([alay_dict[word] if word in alay_dict else word for word in words])

# Tokenisasi
def tokenizing(text):
    return word_tokenize(text)

# Filtering dengan pengecualian kata
def filtering(tokens, exclude_words=None):
    stop_words = set(stopwords.words('indonesian') + stopwords.words('english'))
    
    # Menambahkan kata-kata yang ingin dikecualikan
    if exclude_words is not None:
        stop_words -= set(exclude_words)
    
    # Filter tokens berdasarkan stopwords
    return [word for word in tokens if word not in stop_words and len(word) > 1]

# Stemming
def stemming(tokens):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(token) for token in tokens]

def preprocess_text(text, alay_dict, exclude_words=None):
    print(f"\nTitle       : {text}")
    text_cf = case_folding(text)
    print(f"Case Folding: {text_cf}")
    text_clean = clean_text(text_cf)
    print(f"Cleaning    : {text_clean}")
    text_norm = normalize_text(text_clean, alay_dict)
    print(f"Normalisasi : {text_norm}")
    tokens = tokenizing(text_norm)
    print(f"Tokenizing  : {tokens}")
    tokens_filtered = filtering(tokens, exclude_words=exclude_words)
    print(f"Filtering   : {tokens_filtered}")
    tokens_stemmed = stemming(tokens_filtered)
    print(f"Stemming    : {tokens_stemmed}")
    final_text = ' '.join(tokens_stemmed)
    print(f"Hasil Akhir : {final_text}")
    return final_text

# Fungsi untuk preprocessing dataset dan menyimpannya ke file baru
def preprocess_dataset(input_file, output_file, alay_dict_file, exclude_words=None):
    alay_dict = load_alay_dictionary(alay_dict_file)
    
    df = pd.read_csv(input_file)
    if 'title' in df.columns:
        total_titles = df['title'].count()
        for i, title in enumerate(df['title']):
            print(f"Memproses judul {i + 1}/{total_titles}: {title}")
            df.at[i, 'title'] = preprocess_text(title, alay_dict, exclude_words)
    df.to_csv(output_file, index=False)
    print(f"Dataset telah disimpan di {output_file}")

# Fungsi preprocessing terpisah untuk pickle
def preprocessing_function(text, alay_dict, exclude_words=None):
    return preprocess_text(text, alay_dict, exclude_words)

# Fungsi untuk menyimpan preprocessing pipeline
def save_preprocessing_pipeline(alay_dict_file, output_pipeline_file, exclude_words=None):
    alay_dict = load_alay_dictionary(alay_dict_file)
    pipeline = {
        'alay_dict': alay_dict,
        'preprocess_function': preprocessing_function
    }
    joblib.dump(pipeline, output_pipeline_file)

if __name__ == "__main__":
    # File path untuk dataset dan kamus alay
    input_path = os.path.join('..', 'data', 'hasilCompareCopy.csv') #hasilCompareCopy.csv
    output_path = os.path.join('..', 'data', 'preprocessed.csv') #preprocessed.csv
    alay_dict_path = os.path.join('..', 'data', 'kamusAlay.csv')
    pipeline_output_path = os.path.join('..', 'model', 'preprocessing_pipeline.pkl')
    
    # Kata-kata yang tidak ingin dihapus
    exclude_words = ['ini', 'inilah']

    # Preprocessing dataset dan menyimpannya ke file CSV baru
    preprocess_dataset(input_path, output_path, alay_dict_path, exclude_words=exclude_words)
    
    # Simpan preprocessing pipeline ke file .pkl
    save_preprocessing_pipeline(alay_dict_path, pipeline_output_path, exclude_words=exclude_words)