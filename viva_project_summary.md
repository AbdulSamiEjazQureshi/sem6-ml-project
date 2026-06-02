# Predictive Maintenance ML Project: Viva Explanation and Q&A

## 1. Short Project Explanation

This project is about **predictive maintenance of industrial machines using machine learning**. The aim is to predict whether a machine is likely to fail or not by using sensor readings such as air temperature, process temperature, rotational speed, torque, tool wear, and machine type.

Instead of waiting for a machine to break down, predictive maintenance helps detect failure risk earlier. This can reduce downtime, repair cost, production delay, and safety problems.

The project uses the **AI4I 2020 Predictive Maintenance Dataset**. The dataset has 10,000 machine records. In this dataset, the target value is:

- `0`: Machine is normal / no failure
- `1`: Machine failure risk exists

The project trains and compares three machine learning models:

- Logistic Regression
- K-Nearest Neighbors, also called KNN
- Decision Tree

After training, the best model is saved in `predictive_maintenance_model.pkl`, and it can be used later for prediction on new machine data.

## 2. What The Project Does

The project has four main Python files:

### `load_data.py`

This file loads the dataset from the `data` folder. If the CSV file is not available, it extracts the dataset from the ZIP file. It also cleans column names and shows a basic dataset summary, including rows, columns, failure count, and missing values.

### `train_model.py`

This file prepares the data, trains machine learning models, compares their performance, and saves the best model. It selects important input features:

- Machine type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

The target column is `Target`.

Machine type is categorical, so it is converted into numeric columns using one-hot encoding. Numeric features are scaled for Logistic Regression and KNN using `StandardScaler`.

### `predict_machine.py`

This file loads the saved model and predicts whether a new machine record has failure risk or not. It takes machine readings as input and returns either:

- `Machine failure risk detected`
- `Machine is likely normal`

### `main.py`

This is the main entry point. It prints the project title, shows the dataset summary, and trains the models.

## 3. Why We Chose This Project

We chose this project because predictive maintenance is a practical real-world application of machine learning. In industries, unexpected machine failure can cause financial loss, downtime, and production delays. Machine learning can help identify failure patterns from sensor data and support better maintenance decisions.

This topic is also suitable for a semester ML project because it includes important machine learning steps:

- Dataset loading
- Feature selection
- Data preprocessing
- Categorical encoding
- Train-test split
- Feature scaling
- Model training
- Model comparison
- Model saving
- Prediction on new input

## 4. Why We Chose These Models

### Logistic Regression

We chose Logistic Regression because it is a simple and standard binary classification model. Since the target has only two classes, failure and no failure, Logistic Regression is a suitable baseline model. It is also easy to explain in a viva.

### KNN

We chose KNN because it classifies a new machine by comparing it with similar records in the dataset. If similar machines had failure, KNN may classify the new machine as risky. It is simple and useful for understanding distance-based classification.

### Decision Tree

We chose Decision Tree because it is easy to interpret. It creates decision rules based on features like torque, tool wear, and temperature. For example, it can learn that high tool wear with abnormal torque may indicate failure risk.

## 5. Why We Used These Features

We used sensor and machine condition features because they are directly related to machine health:

- **Air temperature**: High or abnormal air temperature can affect machine operation.
- **Process temperature**: Indicates heat during the actual process.
- **Rotational speed**: Abnormal speed may show operational stress.
- **Torque**: High torque can indicate heavy load or mechanical resistance.
- **Tool wear**: More tool wear means the tool has been used longer and may be closer to failure.
- **Machine type**: Different machine types may behave differently.

We did not use `UDI` and `Product ID` because they are identifiers, not meaningful machine condition features. We also did not use `Failure Type` for prediction because the main target is already `Target`, and using failure type may leak information about the output.

## 6. Important Preprocessing Steps

### One-Hot Encoding

The `Type` column contains categories such as `L`, `M`, and `H`. Machine learning models need numeric values, so we converted this column into separate numeric columns using one-hot encoding.

### Train-Test Split

The dataset is split into training and testing data. The model learns from the training data and is evaluated on the test data. This helps check whether the model can perform on unseen data.

### Stratified Split

The code uses `stratify=target` during splitting. This is important because the dataset is imbalanced. Out of 10,000 records, only 339 records are failure cases. Stratification keeps the class ratio similar in both training and testing data.

### Feature Scaling

Logistic Regression and KNN use scaled data because they are affected by feature ranges. For example, rotational speed has large values while tool wear has smaller values. Scaling helps the model treat features more fairly.

Decision Tree does not require scaling because it splits data based on feature thresholds.

## 7. Limitations We Faced

### 1. Imbalanced Dataset

The biggest limitation is class imbalance. Most records are normal machines, and very few records are failure cases. The dataset has 9,661 normal records and only 339 failure records. Because of this, a model can get high accuracy by mostly predicting "no failure", but it may still miss actual failures.

### 2. Accuracy Alone Is Not Enough

In predictive maintenance, accuracy can be misleading. If the model predicts all machines as normal, it can still look accurate because failures are rare. That is why precision, recall, F1-score, and confusion matrix are also important.

### 3. Limited Features

The project uses available sensor readings only. In real industries, more features may be needed, such as vibration, pressure, humidity, maintenance history, machine age, and real-time sensor logs.

### 4. Dataset Is Not Real-Time

The model is trained on a fixed CSV dataset. It does not connect to live machine sensors. In a real system, we would need real-time data collection and continuous monitoring.

### 5. Simple Models Only

We used basic ML models for learning and explanation. More advanced models like Random Forest, XGBoost, or Gradient Boosting may perform better.

### 6. No User Interface

The project currently runs through Python files. It does not have a dashboard or web interface for maintenance teams.

### 7. Model Selection Uses Accuracy

The saved best model is selected using accuracy. For this problem, recall for failure cases may be more important than accuracy because missing a real failure can be costly.

## 8. What Can Be Improved

### 1. Use Better Evaluation Criteria

Instead of selecting the best model only by accuracy, we can select it using recall or F1-score for the failure class. This would make the model more useful for predictive maintenance.

### 2. Try Advanced Models

We can improve performance by using:

- Random Forest
- XGBoost
- Gradient Boosting
- Support Vector Machine

These models may capture more complex patterns.

### 3. Handle Class Imbalance

We can handle imbalance by using:

- SMOTE oversampling
- Undersampling normal records
- Class weights
- Threshold tuning

This can help the model detect more actual failures.

### 4. Add More Features

More industrial sensor features can improve prediction, such as vibration, pressure, machine age, previous maintenance date, oil temperature, and operating hours.

### 5. Build a Dashboard

A web dashboard can show machine status, failure probability, and maintenance alerts. This would make the system easier for users.

### 6. Show Failure Probability

Instead of only saying normal or failure risk, the system can show probability, such as "Failure risk: 78%". This gives more useful information.

### 7. Deploy The Model

The model can be deployed using Flask, FastAPI, or Streamlit so users can enter machine readings and get predictions easily.

## 9. Likely Viva Questions and Answers

### Q1. What is the main objective of your project?

The main objective is to predict whether an industrial machine is likely to fail or not using machine learning and sensor data.

### Q2. What is predictive maintenance?

Predictive maintenance is a maintenance strategy where data is used to predict machine failure before it happens. It helps reduce unexpected breakdowns and maintenance cost.

### Q3. Which dataset did you use?

We used the AI4I 2020 Predictive Maintenance Dataset. It contains machine sensor readings and a target column that shows whether the machine failed or not.

### Q4. What is the target variable?

The target variable is `Target`. A value of `0` means no failure, and a value of `1` means machine failure.

### Q5. Which features did you use?

We used machine type, air temperature, process temperature, rotational speed, torque, and tool wear.

### Q6. Why did you not use Product ID?

Product ID is only an identifier. It does not directly represent machine condition, so it is not useful for prediction.

### Q7. Why did you use one-hot encoding?

We used one-hot encoding because the `Type` column is categorical. Machine learning models need numeric input, so categories were converted into numeric columns.

### Q8. Why did you use feature scaling?

Feature scaling was used because Logistic Regression and KNN are affected by different feature ranges. Scaling makes the features comparable.

### Q9. Why does Decision Tree not need scaling?

Decision Tree splits data using feature thresholds, so it is not sensitive to feature scale.

### Q10. Which algorithms did you use?

We used Logistic Regression, KNN, and Decision Tree Classifier.

### Q11. Why did you choose Logistic Regression?

We chose it because it is a simple and effective binary classification algorithm. It also works well as a baseline model.

### Q12. Why did you choose KNN?

We chose KNN because it predicts based on similarity between records. Similar machine conditions may have similar failure behavior.

### Q13. Why did you choose Decision Tree?

We chose Decision Tree because it is easy to understand and explain. It creates rules based on the input features.

### Q14. What is the biggest limitation of your project?

The biggest limitation is class imbalance. Failure cases are much fewer than normal cases, so the model may become biased toward predicting no failure.

### Q15. Why is accuracy not enough for this project?

Accuracy is not enough because failures are rare. A model may get high accuracy by predicting most machines as normal, but it may still miss real failures.

### Q16. Which metric is more important for predictive maintenance?

Recall for the failure class is very important because missing an actual failure can be costly and dangerous.

### Q17. What is a confusion matrix?

A confusion matrix shows correct and incorrect predictions for each class. It helps us see false positives and false negatives.

### Q18. What is a false negative in this project?

A false negative means the machine is actually going to fail, but the model predicts it as normal. This is dangerous in predictive maintenance.

### Q19. What is a false positive in this project?

A false positive means the model predicts failure risk, but the machine is actually normal. This may cause unnecessary maintenance.

### Q20. Which is worse: false positive or false negative?

False negative is usually worse because it means a real failure is missed. A false positive may only cause extra checking, but a false negative can cause breakdown.

### Q21. What is class imbalance?

Class imbalance means one class has many more records than the other. In this dataset, normal machines are much more common than failed machines.

### Q22. How can you solve class imbalance?

We can use SMOTE, undersampling, oversampling, class weights, or threshold tuning.

### Q23. Why did you save the model?

We saved the model so it can be reused for future predictions without training again.

### Q24. Which library is used to save the model?

We used `joblib` to save the trained model, scaler, feature columns, model name, and accuracy.

### Q25. What does `predict_machine.py` do?

It loads the saved model, prepares new machine data in the same format as training data, and predicts whether failure risk exists.

### Q26. Why is it important to use the same columns during prediction?

The model was trained on specific columns. During prediction, the new input must match the same column structure, otherwise the model may give wrong results or fail.

### Q27. What is the role of `StandardScaler`?

`StandardScaler` transforms features so they have a similar scale. This helps algorithms like Logistic Regression and KNN.

### Q28. What changes would you make in the future?

We would use advanced models, handle class imbalance better, select the best model using recall or F1-score, add a dashboard, and connect the system with real-time sensor data.

### Q29. How is this project useful in real life?

It can help industries detect machine failure risk early, schedule maintenance, reduce downtime, and avoid unexpected breakdowns.

### Q30. How would you explain this project in two minutes?

Our project predicts machine failure using machine learning. We used the AI4I predictive maintenance dataset, which contains machine sensor readings like temperature, rotational speed, torque, and tool wear. We trained Logistic Regression, KNN, and Decision Tree models to classify whether a machine is normal or at failure risk. We handled categorical data using one-hot encoding and scaled numeric data for models that need scaling. The best model is saved and used later for prediction. The main limitation is class imbalance because failure records are very few compared to normal records. In the future, we can improve it using advanced models, better imbalance handling, a dashboard, and real-time sensor data.

## 10. Simple Explanation For Sir

Sir, this project applies machine learning to predictive maintenance. We used machine sensor readings to predict whether a machine may fail. The dataset contains 10,000 records, and each record has values like temperature, speed, torque, tool wear, and machine type. We trained three models: Logistic Regression, KNN, and Decision Tree. We compared them using accuracy, confusion matrix, and classification report. The best model is saved and can be used to predict new machine data.

The main challenge was that the dataset is imbalanced, because failure cases are very few. So, accuracy alone is not enough. In a real predictive maintenance system, recall is very important because missing an actual failure is more costly than raising a false alarm. Future improvements include advanced models, better handling of imbalance, real-time sensor input, and a user dashboard.

