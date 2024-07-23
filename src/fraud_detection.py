import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
    df["Time"] = scaler.fit_transform(df["Time"].values.reshape(-1, 1))
    df["Class"] = df["Class"].replace({0: 1, 1: -1})
    X = df.drop(["Class"], axis=1)
    y = df["Class"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_isolation_forest(X_train):
    iso_forest = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    iso_forest.fit(X_train)
    return iso_forest

def evaluate_model(model, X_test, y_test):
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data("../data/creditcard.csv")
    model = train_isolation_forest(X_train)
    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
    print("Accuracy Score:", accuracy)
    print("Precision Score:", precision)
    print("Recall Score:", recall)
    print("F1 Score:", f1)
