import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from load_data import load_dataset


MODEL_FILE = "predictive_maintenance_model.pkl"


def prepare_data(data):
    features = data[
        [
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]
    ]
    target = data["Target"]

    features = pd.get_dummies(features, columns=["Type"], drop_first=False)
    return features, target


def train_models():
    data = load_dataset()
    features, target = prepare_data(data)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    }

    best_model_name = ""
    best_model = None
    best_accuracy = 0

    for name, model in models.items():
        if name == "Decision Tree":
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
        else:
            model.fit(x_train_scaled, y_train)
            predictions = model.predict(x_test_scaled)

        accuracy = accuracy_score(y_test, predictions)
        print("\nModel:", name)
        print("Accuracy:", round(accuracy, 4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        print("Classification Report:")
        print(classification_report(y_test, predictions, zero_division=0))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model

    saved_data = {
        "model": best_model,
        "model_name": best_model_name,
        "accuracy": best_accuracy,
        "scaler": scaler,
        "columns": list(features.columns),
    }
    joblib.dump(saved_data, MODEL_FILE)

    print("\nBest Model:", best_model_name)
    print("Best Accuracy:", round(best_accuracy, 4))
    print("Saved model file:", MODEL_FILE)


if __name__ == "__main__":
    train_models()
