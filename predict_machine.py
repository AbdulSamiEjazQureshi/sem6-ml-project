import joblib
import pandas as pd

from train_model import MODEL_FILE


def predict_failure(machine_data):
    saved_data = joblib.load(MODEL_FILE)
    model = saved_data["model"]
    model_name = saved_data["model_name"]
    scaler = saved_data["scaler"]
    columns = saved_data["columns"]

    data = pd.DataFrame([machine_data])
    data = pd.get_dummies(data, columns=["Type"], drop_first=False)
    data = data.reindex(columns=columns, fill_value=0)

    if model_name == "Decision Tree":
        prediction = model.predict(data)[0]
    else:
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)[0]

    if prediction == 1:
        return "Machine failure risk detected"
    return "Machine is likely normal"


if __name__ == "__main__":
    sample_machine = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
    }

    result = predict_failure(sample_machine)
    print(result)
