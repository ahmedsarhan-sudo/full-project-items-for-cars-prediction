🚗 Car Price Prediction — From Raw Data to Deployed ML System

A complete end-to-end Machine Learning project — from raw data exploration to a fully deployed production-ready API and interactive dashboard.
📖 Project Journey

This project started with a dataset containing car specifications and their prices.
What began as simple data analysis evolved into a complete Machine Learning system — including preprocessing, modeling, evaluation, deployment, and visualization.

This was not just model training — it was a full real-world ML workflow.
📖 Project Journey

This project started with a dataset containing car specifications and their prices.
What began as simple data analysis evolved into a complete Machine Learning system — including preprocessing, modeling, evaluation, deployment, and visualization.

This was not just model training — it was a full real-world ML workflow.
📊 Phase 1 — Data Understanding

The first step was deeply understanding the dataset.

Steps taken:

Explored every column and understood its purpose

Checked data types for correctness

Analyzed missing values (NaNs)

Calculated null percentages

Removed duplicate rows

Fixed incorrect data types

Before moving to modeling, I ensured the dataset was clean and logically structured.
📈 Phase 2 — Exploratory Data Analysis (EDA)

I created multiple visualizations to better understand:

Manufacturers distribution

Price trends

Feature correlations

Impact of features on the target variable (price)

Through visualization, I gained deeper insights into:

Which features significantly affect price

Potential noise in the dataset

Patterns that weren't obvious from raw numbers

This phase gave me confidence to proceed with modeling.
🧹 Phase 3 — Data Cleaning & Outlier Detection

I created a dedicated preprocessing module where I:

Detected outliers

Visualized their effect on the target variable

Removed harmful outliers

Re-validated distributions

After cleaning, I updated my main pipeline to include all improvements.

This step significantly improved model stability and performance.
⚙️ Phase 4 — Feature Engineering & Encoding

Features were divided into:

Label Encoded columns

One-Hot Encoded columns

Scaled numerical columns

I built a structured ML pipeline that included:

Encoding

Scaling

Model training

I experimented with two encoding strategies:

One-Hot Encoding

Target Encoding
🤖 Phase 5 — Model Experimentation

I trained multiple models and compared performance.

🔹 Linear Regression

Accuracy: ~30%

🔹 Random Forest Regressor

Accuracy: ~60%

Then I tested Target Encoding:

Linear Regression accuracy improved

But Random Forest with One-Hot Encoding performed better overall
🔁 Cross Validation

To ensure model stability and avoid overfitting, I performed:

Cross-validation on Linear Regression

Cross-validation on Random Forest

Compared both encoding strategies

After thorough experimentation, the final chosen model was:

🎯 Random Forest Regressor with One-Hot Encoding
📈 Final Model Performance

After full preprocessing, feature engineering, encoding optimization, and validation:

🚀 Final Accuracy: 81%

This improvement came from:

Proper data cleaning

Outlier removal

Feature engineering

Encoding experimentation

Cross validation

Model comparison

The result was achieved through systematic experimentation, not trial-and-error.
📊 Model Visualization

I created visualizations to:

Compare predicted vs actual values

Analyze residuals

Understand model fit behavior

Evaluate performance graphically

These visualizations confirmed the model generalizes well.
🌐 Phase 6 — Production Deployment

After finalizing the model, I transformed it into a real application:

Backend API

Built using FastAPI

Handles prediction requests

Loads trained model and preprocessing objects

Returns predicted price via API endpoint

Deployment

Deployed backend as a serverless function

Fully accessible via public API

Dashboard

Built a Streamlit dashboard

Integrated visualizations

Showcased analysis and model insights

Deployed separately for interactive usage

This turned the project into a production-ready ML system.
🏗️ Tech Stack

Python

Pandas

NumPy

Scikit-learn

FastAPI

Streamlit

Serverless Deployment
🎯 What This Project Represents

This project demonstrates:

End-to-end Machine Learning workflow

Real-world data cleaning

Outlier handling

Feature engineering

Model experimentation

Cross-validation

Backend API development

Deployment experience

Dashboard creation

Production debugging

It required multiple days of iteration, debugging, and refinement — but resulted in a complete, deployable ML system.

👨‍💻 Author

Ahmed Sarhan

Machine Learning Engineer (In Progress 🚀)

⭐ If you found this project interesting, feel free to star the repository!
