import os
import sys
import joblib
import pandas as pd
import requests
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from flask import Flask, render_template, request

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import load_alay_dictionary, preprocess_text

app = Flask(__name__)
API_KEY = 'AIzaSyDMB88euf37_JmSwCNXTWpbM1-Kk4ySafw'
URL = f'https://youtube.googleapis.com/youtube/v3/videos?part=snippet&chart=mostPopular&regionCode=ID&maxResults=50&key={API_KEY}'

# Definisi ulang preprocessing_function
def preprocessing_function(text, alay_dict, exclude_words=None):
    return preprocess_text(text, alay_dict, exclude_words)

# Load semua model dan komponen
def load_models():
    try:
        pipeline = joblib.load(os.path.join('model', 'preprocessing_pipeline.pkl'))
        tfidf_vectorizer = joblib.load(os.path.join('model', 'tfidf_vectorizer.pkl'))
        svm_model = joblib.load(os.path.join('model', 'svm_model.pkl'))
        return pipeline, tfidf_vectorizer, svm_model
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

pipeline, tfidf_vectorizer, svm_model = load_models()
alay_dict = pipeline['alay_dict'] if pipeline else {}

# Ambil video trending dari YouTube
def fetch_trending_videos():
    response = requests.get(URL)
    data = response.json()
    return data.get('items', [])

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if not pipeline or not tfidf_vectorizer or not svm_model:
            return "Error loading models. Check console for details."

        try:
            judul = request.form['judul']
            preprocessed = preprocess_text(judul, alay_dict)
            vector = tfidf_vectorizer.transform([preprocessed]).toarray()
            label = svm_model.predict(vector)[0]
            hasil = "Clickbait" if label == 1 else "Non-Clickbait"
            return render_template('home.html', hasil=hasil, judul_asli=judul)
        except Exception as e:
            print(f"Error during manual prediction: {e}")
            return f"An error occurred: {e}"

    return render_template('home.html')

@app.route('/detect')
def detect_clickbait():
    if not pipeline or not tfidf_vectorizer or not svm_model:
        return "Error loading models. Check console for details."

    try:
        videos = fetch_trending_videos()
        processed_videos = []

        for video in videos:
            title = video['snippet']['title']
            lang = detect_language(title)
            preprocessed_title = preprocess_text(title, alay_dict)

            if lang in ['ko']:
                label_display = "uknown"
            else:
                title_vector = tfidf_vectorizer.transform([preprocessed_title]).toarray()
                label = svm_model.predict(title_vector)[0]
                label_display = "Clickbait" if label == 1 else "Non-Clickbait"

            processed_videos.append({
                'id': video['id'],
                'title': title,
                'processed_title': preprocessed_title,
                'label': label_display
            })

        return render_template('index.html', videos=processed_videos)
    except Exception as e:
        print(f"Error occurred while processing videos: {e}")
        return f"An error occurred while processing videos: {e}"

if __name__ == '__main__':
    app.run(debug=True)
