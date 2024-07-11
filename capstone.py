import streamlit as st
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Fungsi untuk memuat data
def load_data():
    dir = 'hungarian.data'
    with open(dir, encoding='Latin1') as file:
        lines = [line.strip() for line in file]
    data = itertools.takewhile(
        lambda x: len(x) == 76,
        (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
    )
    df = pd.DataFrame.from_records(data)
    df.replace('-9', np.nan, inplace=True)
    df.drop(columns=df.columns[-1], inplace=True)
    selected_columns = [2, 3, 8, 9, 11, 15, 18, 31, 37, 39, 40, 43, 50, 57]
    df = df.iloc[:, selected_columns]
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                  'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    df = df.astype(float)
    df.fillna(df.mean(), inplace=True)
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

# Fungsi untuk membangun dan mengevaluasi model
def build_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel='linear'),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "XGBoost": XGBClassifier(objective="binary:logistic", random_state=42)
    }
    
    results = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((model_name, accuracy))
    return results

# Fungsi utama Streamlit
def main():
    st.title("Heart Disease Classification")
    
    # Memuat data
    df = load_data()
    
    st.write("### DataFrame")
    st.write(df.head())
    
    st.write("### Korelasi Matriks")
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    st.pyplot(plt)
    
    # Memisahkan fitur (X) dan label (y)
    X = df.drop(columns=['num'])
    y = df['num']
    
    # Plot distribusi kelas sebelum oversampling
    st.write("### Distribusi Kelas Sebelum Oversampling")
    fig, ax = plt.subplots()
    y.value_counts().sort_index().plot(kind='bar', title='Distribusi Kelas Sebelum Oversampling', ax=ax)
    plt.xlabel('Kelas')
    plt.ylabel('Frekuensi')
    st.pyplot(fig)
    
    # Menerapkan SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Plot distribusi kelas setelah oversampling
    st.write("### Distribusi Kelas Setelah Oversampling")
    fig, ax = plt.subplots()
    y_res.value_counts().sort_index().plot(kind='bar', title='Distribusi Kelas Setelah Oversampling', ax=ax)
    plt.xlabel('Kelas')
    plt.ylabel('Frekuensi')
    st.pyplot(fig)
    
    # Normalisasi atau standarisasi fitur
    scaler = StandardScaler()
    X_res = scaler.fit_transform(X_res)
    
    # Membagi data menjadi set pelatihan dan set pengujian
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    # Membangun dan mengevaluasi model
    results = build_and_evaluate_models(X_train, y_train, X_test, y_test)
    
    # Menampilkan hasil evaluasi model
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
    st.write("### Akurasi Model")
    st.write(results_df)
    
    # Plot akurasi model
    st.write("### Akurasi Score Barplot")
    fig, ax = plt.subplots()
    sns.barplot(x='Model', y='Accuracy', data=results_df, ax=ax)
    plt.xlabel("Model")
    plt.ylabel("Accuracy score")
    st.pyplot(fig)

if __name__ == '__main__':
    main()
