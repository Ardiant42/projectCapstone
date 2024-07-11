import streamlit as st
import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Fungsi untuk memuat data
def load_data():
    with open('hungarian.data', encoding='Latin1') as file:
        lines = [line.strip() for line in file]
    data = []
    for i in range(0, len(lines), 10):
        data.append(' '.join(lines[i:i + 10]).split())
    df = pd.DataFrame(data)
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
    df['num'] = df['num'].astype(int)  # Mengonversi 'num' ke tipe data integer
    return df

# Fungsi untuk melatih model
def train_models(df):
    X = df.drop(columns=['num'])
    y = df['num']
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    scaler = StandardScaler()
    X_res = scaler.fit_transform(X_res)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    # Melatih model KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    # Melatih model XGBoost
    xgb_model = XGBClassifier(objective="binary:logistic", random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Melatih model Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    
    return scaler, knn, xgb_model, dt, X_test, y_test

# Fungsi untuk membuat prediksi
def make_prediction(model, scaler, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    return model.predict(input_data)

# Main function
def main():
    st.title("Heart Disease Prediction")
    
    # Memuat data dan melatih model
    df = load_data()
    scaler, knn, xgb_model, dt, X_test, y_test = train_models(df)
    
    st.write("### Input Attributes")
    age = st.number_input("Usia (age)", min_value=0, max_value=120, value=25)
    sex = st.selectbox("Jenis Kelamin (sex) (1-laki-laki 0-perempuan)", options=['female', 'male'])
    cp = st.selectbox("Tipe nyeri dada (cp)", options=['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
    trestbps = st.number_input("Tekanan darah saat ini (trestbps)", min_value=0, max_value=300, value=120)
    chol = st.number_input("Kolesterol (chol)", min_value=0, max_value=600, value=200)
    fbs = st.selectbox("Kadar Gula Darah > 120 mg/dl (fbs)", options=['false', 'true'])
    restecg = st.selectbox("Hasil Elektrokardiografi saat istirahat (restecg)", options=['normal', 'having ST-T wave abnormality', 'showing probable or definite left ventricular hypertrophy'])
    thalach = st.number_input("Detak jantung maksimum (thalach)", min_value=0, max_value=250, value=150)
    exang = st.selectbox("Induksi Angina oleh olahraga (exang)", options=['no', 'yes'])
    oldpeak = st.number_input("Induksi Depresi ST oleh olahraga (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Kemiringan ST Puncak saat olahraga (slope)", options=['upsloping', 'flat', 'downsloping'])
    ca = st.number_input("Pembuluh darah yang diwarnai Fluoroskopi (ca)", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Kondisi Talasemia (thal)", options=['normal', 'fixed defect', 'reversible defect'])
    
    # Mengonversi input_data ke format numerik yang sesuai
    sex = 1 if sex == 'male' else 0
    cp_map = {'typical angina': 0, 'atypical angina': 1, 'non-anginal pain': 2, 'asymptomatic': 3}
    cp = cp_map[cp]
    fbs = 1 if fbs == 'true' else 0
    restecg_map = {'normal': 0, 'having ST-T wave abnormality': 1, 'showing probable or definite left ventricular hypertrophy': 2}
    restecg = restecg_map[restecg]
    exang = 1 if exang == 'yes' else 0
    slope_map = {'upsloping': 0, 'flat': 1, 'downsloping': 2}
    slope = slope_map[slope]
    thal_map = {'normal': 1, 'fixed defect': 2, 'reversible defect': 3}
    thal = thal_map[thal]
    
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    
    if st.button("Predict with KNN"):
        result_knn = make_prediction(knn, scaler, input_data)
        st.write(f"KNN Prediction: {'Heart Disease' if result_knn[0] > 0 else 'No Heart Disease'}")
        
    if st.button("Predict with XGBoost"):
        result_xgb = make_prediction(xgb_model, scaler, input_data)
        st.write(f"XGBoost Prediction: {'Heart Disease' if result_xgb[0] > 0 else 'No Heart Disease'}")
    
    if st.button("Predict with Decision Tree"):
        result_dt = make_prediction(dt, scaler, input_data)
        st.write(f"Decision Tree Prediction: {'Heart Disease' if result_dt[0] > 0 else 'No Heart Disease'}")
        
        # Menampilkan plot dari Decision Tree
        st.write("### Decision Tree Structure")
        st.set_option('deprecation.showPyplotGlobalUse', False)  # Disable deprecated warning
        plot_tree(dt, filled=True)
        st.pyplot()
    
if __name__ == '__main__':
    main()
