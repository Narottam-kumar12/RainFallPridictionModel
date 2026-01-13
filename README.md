# 🌧️ Rainfall Prediction using Machine Learning (Australia Weather Dataset)

## 📌 Project Overview

This project aims to predict **whether it will rain tomorrow** using historical weather data from Australia. The problem is framed as a **binary classification task**, where the target variable is **RainTomorrow (Yes/No)**.

The notebook demonstrates an **end-to-end data science pipeline**, covering data preprocessing, class imbalance handling, feature engineering, exploratory data analysis (EDA), machine learning modeling, and performance evaluation.

---

## 🎯 Problem Statement

Accurate rainfall prediction is essential for agriculture, disaster management, and water resource planning. Weather datasets are often noisy, imbalanced, and contain missing values. This project addresses these challenges using robust preprocessing and multiple machine learning models.

---

## 🗂️ Dataset Information

* **Dataset:** Australian Weather Dataset (`weatherAUS.csv`)
* **Target Variable:** `RainTomorrow`

  * 1 → Rain Tomorrow
  * 0 → No Rain Tomorrow

### Key Features Include:

* Temperature (MinTemp, MaxTemp, Temp9am, Temp3pm)
* Rainfall, Evaporation
* Humidity, Pressure, Cloud Cover
* Wind Speed & Direction
* RainToday (binary)

---

## ⚠️ Class Imbalance

Initial analysis shows:

* ~78% samples → No Rain Tomorrow (0)
* ~22% samples → Rain Tomorrow (1)

➡️ The dataset is **highly imbalanced**, which can bias model predictions.

### Solution Applied:

* **Random Oversampling** of the minority class to balance the dataset

---

## 🧹 Data Preprocessing & Feature Engineering

* Converted categorical target variables (`RainToday`, `RainTomorrow`) into binary values
* Handled missing values using:

  * Mode imputation for categorical features
  * **MICE (Iterative Imputer)** for numerical features
* Label Encoding for categorical variables
* Outlier detection and removal using **IQR method**

---

## 🔍 Exploratory Data Analysis (EDA)

* Missing value heatmaps
* Feature correlation analysis using heatmaps
* Pair plots for strongly correlated features
* Identified strong correlations such as:

  * Humidity9am ↔ Humidity3pm
  * MaxTemp ↔ Temp3pm
  * Pressure9am ↔ Pressure3pm

EDA helped uncover meaningful relationships between weather variables and rainfall.

---

## 🧠 Machine Learning Models Implemented

The following models were trained and evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Multi-layer Perceptron (Neural Network)
* XgBoost
* Light GBM
* CatBoost
* Ensemble
