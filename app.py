import streamlit as st
import pandas as pd
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load and preprocess data
def load_data(filepath):
    df = pd.read_csv(filepath)
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
    df["Time"] = scaler.fit_transform(df["Time"].values.reshape(-1, 1))
    df["Class"] = df["Class"].replace({0: 1, 1: -1})
    return df

# Train the Isolation Forest model
def train_model(X):
    model = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    model.fit(X)
    return model

# Streamlit app
def main():
    st.title("Fraud Spotter")
    st.markdown("""
    **Fraud Spotter** is a tool designed to identify fraudulent credit card transactions using machine learning.
    This project utilizes the following tools and strategies:
    - **Pandas** for data manipulation and analysis.
    - **Scikit-Learn** for implementing the Isolation Forest algorithm for anomaly detection.
    - **Streamlit** for creating an interactive web application.
    - **Altair** for visualizing the results.
    """)

    st.markdown("""
    ### Dataset Description
    The dataset used in this project is the Credit Card Fraud Detection dataset from Kaggle.
    - It contains transactions made by credit cards in September 2013 by European cardholders.
    - This dataset presents transactions that occurred over two days, with 492 frauds out of 284,807 transactions.
    - The dataset is highly unbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.
    - Features `V1`, `V2`, ... `V28` are the principal components obtained with PCA, while `Time` and `Amount` are not transformed.
    - `Class` is the response variable and it takes the value 1 in cases of fraud and 0 otherwise.
    """)

    # File upload
    uploaded_file = st.file_uploader("Upload Credit Card Transactions CSV", type=["csv"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        X = data.drop(["Class"], axis=1)
        y = data["Class"]

        # Train model
        model = train_model(X)

        # Predict and display results
        predictions = model.predict(X)
        data["Predictions"] = predictions
        st.write("Prediction Results:")
        st.write(data)

        # Display evaluation metrics
        st.subheader("Evaluation Metrics")
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        f1 = f1_score(y, predictions)

        st.write(f"**Accuracy**: {accuracy:.2f}")
        st.write("Accuracy is the ratio of correctly predicted observations to the total observations.")
        
        st.write(f"**Precision**: {precision:.2f}")
        st.write("Precision is the ratio of correctly predicted positive observations to the total predicted positives.")
        
        st.write(f"**Recall**: {recall:.2f}")
        st.write("Recall is the ratio of correctly predicted positive observations to all observations in actual class.")
        
        st.write(f"**F1 Score**: {f1:.2f}")
        st.write("F1 Score is the weighted average of Precision and Recall, balancing both concerns.")

        # Visualize the results using Altair
        st.subheader("Predictions Visualization")
        chart = alt.Chart(data).mark_point().encode(
            x='Time',
            y='Amount',
            color=alt.condition(
                alt.datum.Predictions == -1, 
                alt.value('red'),     # The color for outliers
                alt.value('blue')     # The color for inliers
            ),
            tooltip=['Time', 'Amount', 'Predictions']
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        # Display the distribution of anomalies
        st.subheader("Distribution of Anomalies")
        anomaly_count = data['Predictions'].value_counts().reset_index()
        anomaly_count.columns = ['Prediction', 'Count']
        anomaly_chart = alt.Chart(anomaly_count).mark_bar().encode(
            x='Prediction:N',
            y='Count:Q',
            color='Prediction:N'
        )
        st.altair_chart(anomaly_chart, use_container_width=True)

if __name__ == "__main__":
    main()
