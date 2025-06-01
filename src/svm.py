import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

def load_data(data_file, tfidf_file):
    df = pd.read_csv(data_file)
    tfidf_matrix = pd.read_csv(tfidf_file).values

    if 'label' not in df.columns or df['label'].isnull().any():
        raise ValueError("Kolom 'label' tidak ditemukan atau mengandung nilai kosong!")

    return tfidf_matrix, df['label'], df

def train_svm(X_train, y_train):
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model

# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)

#     conf_matrix = confusion_matrix(y_test, y_pred)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, zero_division=1)
#     recall = recall_score(y_test, y_pred, zero_division=1)
#     f1 = f1_score(y_test, y_pred, zero_division=1)

#     return y_pred, conf_matrix, accuracy, precision, recall, f1

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Hitung akurasi manual dari confusion matrix
    correct = conf_matrix[0][0] + conf_matrix[1][1]  # TP + TN
    total = conf_matrix.sum()
    accuracy = correct / total

    return y_pred, conf_matrix, accuracy

def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

def print_label_stats(df):
    label_counts = df['label'].value_counts()
    clickbait_count = label_counts.get(1, 0)
    non_clickbait_count = label_counts.get(0, 0)

    print(f"\nTotal Clickbait     : {clickbait_count}")
    print(f"Total Non-Clickbait : {non_clickbait_count}")

def save_predictions_to_csv(df, y_test, y_pred, output_path):
    # Ambil kembali data test (judul dan label aktual)
    test_indices = y_test.index
    df_test = df.loc[test_indices].copy()
    df_test['label_aktual'] = y_test.values
    df_test['label_prediksi'] = y_pred
    df_test[['title', 'label_aktual', 'label_prediksi']].to_csv(output_path, index=False)
    print(f"Hasil prediksi disimpan di {output_path}")

if __name__ == "__main__":
    # Path file
    data_file = os.path.join('..', 'data', 'preprocessed.csv')
    tfidf_file = os.path.join('..', 'data', 'tfidf.csv')
    model_file = os.path.join('..', 'model', 'svm_model.pkl')
    prediction_output_file = os.path.join('..', 'data', 'prediksi_testing.csv')

    # Load data
    X, y, df = load_data(data_file, tfidf_file)

    # Bagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Latih model
    print("Melatih model SVM...")
    svm_model = train_svm(X_train, y_train)

    # Simpan model
    save_model(svm_model, model_file)
    print(f"Model SVM disimpan di {model_file}")

    # Evaluasi
    # y_pred, conf_matrix, accuracy, precision, recall, f1 = evaluate_model(svm_model, X_test, y_test)
    y_pred, conf_matrix, accuracy = evaluate_model(svm_model, X_test, y_test)

    # Statistik label
    print_label_stats(df)

    # Print hasil evaluasi
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print(f"\nAccuracy : {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall   : {recall}")
    # print(f"F1 Score : {f1}")

    # Simpan hasil prediksi ke CSV
    save_predictions_to_csv(df, y_test, y_pred, prediction_output_file)
