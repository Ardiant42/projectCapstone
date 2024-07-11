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

# Function to load data
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

# Function to build and evaluate models
def build_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel='linear'),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(objective="binary:logistic", random_state=42)
    }
    
    results = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((model_name, accuracy))
    return results

# Main function for Streamlit app
def main():
    st.title("Heart Disease Classification")
    st.markdown("### Project Capstone Heart Disease")
    st.markdown("Dataset: Hungarian Dataset")
    st.markdown("[Source: UCI Heart Disease Data](http://archive.ics.uci.edu/dataset/45/heart+disease)")
    
    # Load data
    df = load_data()
    
    # Display data table
    st.subheader("Data Overview")
    st.write(df.head())
    
    # Correlation matrix
    st.subheader("Correlation Matrix")
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))  # Create a figure and axis object
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix')  # Set title for the plot
    st.pyplot(fig)  # Display the plot
    
    # Class distribution before and after oversampling
    st.subheader("Class Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].set_title("Before Oversampling")
    df['num'].value_counts().sort_index().plot(kind='bar', ax=axes[0])
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Frequency")
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(df.drop(columns=['num']), df['num'])
    pd.Series(y_res).value_counts().sort_index().plot(kind='bar', ax=axes[1])
    axes[1].set_title("After Oversampling")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Frequency")
    
    st.pyplot(fig)
    
    # Normalization or standardization of features
    scaler = StandardScaler()
    X_res = scaler.fit_transform(X_res)
    
    # Train-test split after oversampling
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    # Build and evaluate models
    st.subheader("Model Evaluation")
    results = build_and_evaluate_models(X_train, y_train, X_test, y_test)
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
    
    # Display accuracy scores
    st.write(results_df)
    
    # Plot accuracy scores
    st.subheader("Accuracy Scores")
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure for the bar plot
    sns.barplot(x='Model', y='Accuracy', data=results_df, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # Rotate x-axis labels for better readability
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    st.pyplot(fig)  # Display the plot

if __name__ == '__main__':
    main()
