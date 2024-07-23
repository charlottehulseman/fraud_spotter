# Fraud Spotter

**Fraud Spotter** is a tool designed to identify fraudulent credit card transactions using machine learning. This project utilizes the Isolation Forest algorithm for anomaly detection and provides an interactive web application to visualize and evaluate the results.

## Table of Contents
- [Project Description](#project-description)
- [Tools and Strategies](#tools-and-strategies)
- [Dataset Description](#dataset-description)
- [Setup Instructions](#setup-instructions)
- [Running the App](#running-the-app)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Description
Fraud Spotter is designed to identify fraudulent credit card transactions using an unsupervised machine learning approach. It leverages the Isolation Forest algorithm to detect anomalies in transaction data. The project includes a web interface built with Streamlit for easy interaction and visualization of the results.

## Tools and Strategies
This project utilizes the following tools and strategies:
- **Pandas** for data manipulation and analysis.
- **Scikit-Learn** for implementing the Isolation Forest algorithm for anomaly detection.
- **Streamlit** for creating an interactive web application.
- **Altair** for visualizing the results.

## Dataset Description
The dataset used in this project is the Credit Card Fraud Detection dataset from Kaggle.
- It contains transactions made by credit cards in September 2013 by European cardholders.
- The dataset includes transactions that occurred over two days, with 492 frauds out of 284,807 transactions.
- The dataset is highly unbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.
- Features `V1`, `V2`, ... `V28` are the principal components obtained with PCA, while `Time` and `Amount` are not transformed.
- `Class` is the response variable and it takes the value 1 in cases of fraud and 0 otherwise.

## Setup Instructions
1. **Clone the Repository:**
    ```sh
    git clone https://github.com/yourusername/fraud_spotter.git
    cd fraud_spotter
    ```

2. **Install the Required Dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Download the Dataset:**
    - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` directory.

## Running the App
1. **Start the Streamlit App:**
    ```sh
    streamlit run app.py
    ```

2. **Upload the `creditcard.csv` File:**
    - Use the file uploader in the app to upload the `creditcard.csv` file and view the anomaly detection results.

## Usage
- **Evaluation Metrics:**
  - **Accuracy:** The ratio of correctly predicted observations to the total observations.
  - **Precision:** The ratio of correctly predicted positive observations to the total predicted positives.
  - **Recall:** The ratio of correctly predicted positive observations to all observations in actual class.
  - **F1 Score:** The weighted average of Precision and Recall.

- **Visualization:**
  - Interactive visualization of predictions using Altair.
  - Distribution of anomalies using bar charts.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
