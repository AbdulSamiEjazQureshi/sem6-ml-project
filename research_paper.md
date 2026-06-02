# Predictive Maintenance of Industrial Machines Using Machine Learning

## Students

Abdul Sami  
Roll No: 2023-uam-1875

Hasham Rao  
Roll No: 2023-uam-1923

## Abstract

Predictive maintenance is an important machine learning application in modern industry. Industrial machines generate sensor readings such as temperature, rotational speed, torque, and tool wear. These readings can help identify whether a machine is likely to fail. This research paper presents a machine learning approach for predicting machine failure using the AI4I 2020 predictive maintenance dataset. The project compares Logistic Regression, K-Nearest Neighbors, and Decision Tree models. The main goal is to detect failure risk early so that maintenance can be planned before breakdown occurs.

## Keywords

Predictive Maintenance, Machine Learning, Industrial Machines, Classification, Logistic Regression, KNN, Decision Tree

## 1. Introduction

Industrial machines are used in manufacturing, production, and engineering systems. When a machine fails unexpectedly, it can cause downtime, financial loss, production delay, and safety issues. Traditional maintenance methods are reactive or preventive. Reactive maintenance repairs the machine after failure. Preventive maintenance follows a fixed schedule, but it may repair machines too early or too late.

Predictive maintenance uses machine data to predict failure before it happens. Machine learning models can study previous machine records and learn patterns related to failure. This makes maintenance more efficient and helps industries reduce unexpected breakdowns.

## 2. Problem Statement

The problem is to predict whether an industrial machine will fail based on sensor readings. The target variable is binary. A value of 0 means no failure, and a value of 1 means machine failure. The research question is:

Which machine learning model can predict industrial machine failure more accurately using sensor data?

## 3. Dataset Description

The dataset used in this project is the AI4I 2020 predictive maintenance dataset. The dataset is stored in the `notes` folder as `predictive_maintenance.csv`. A ZIP version of the dataset is also available in the same folder.

The dataset contains machine records with the following important features:

- Machine type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Failure type
- Target value

The target column is used to train the model. The failure type column gives extra information about the type of machine failure.

## 4. Methodology

The project follows these steps:

1. Load the dataset from the `notes` folder.
2. Clean column names and inspect the dataset.
3. Select useful input features.
4. Convert machine type into numeric columns.
5. Split the data into training and testing sets.
6. Scale numeric values for Logistic Regression and KNN.
7. Train multiple machine learning models.
8. Compare model performance using accuracy, confusion matrix, precision, recall, and F1-score.

## 5. Algorithms Used

### 5.1 Logistic Regression

Logistic Regression is a classification algorithm used for binary prediction. It predicts the probability that a machine belongs to the failure class.

### 5.2 K-Nearest Neighbors

KNN classifies a new machine record by comparing it with nearby records in the dataset. It works well when similar machines have similar failure behavior.

### 5.3 Decision Tree

Decision Tree creates simple rules from the dataset. For example, it may split data based on tool wear, torque, or temperature. It is easy to understand and explain.

## 6. Evaluation Metrics

The models are evaluated using:

- Accuracy
- Confusion matrix
- Precision
- Recall
- F1-score

In predictive maintenance, recall is important because missing a real failure can be costly. A false alarm is usually less dangerous than a missed machine failure.

## 7. Expected Result

The expected result is a trained model that can classify machine records as normal or failure-risk. The best model is saved in the project directory as `predictive_maintenance_model.pkl` after training.

## 8. Conclusion

This project shows how machine learning can be used for predictive maintenance of industrial machines. By using sensor data, the system can predict machine failure before it happens. This helps industries reduce downtime and improve maintenance planning. Logistic Regression, KNN, and Decision Tree provide simple and understandable models for this classification problem.

## 9. Future Work

Future improvements can include Random Forest, XGBoost, deep learning, real-time sensor data, and a dashboard for maintenance alerts.
