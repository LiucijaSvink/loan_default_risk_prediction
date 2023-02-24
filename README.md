## Introduction
The goal of this project is to create a tool based on a Machine Learning model that could be used by retail banks to evaluate the probability of a loan default. The proposed usage of the model is adjusting the interest rate based on the default probability. This way, interest rates tailored for every client can be derived. In addition, one of the variables, coming from external sources is predicted to see whether it can be estimated from the data that is present in the Home Credit database.

## Data
A dataset from [Home Credit Group](https://www.kaggle.com/competitions/home-credit-default-risk) was used.

## Content
Notebooks
- **Home credit default risk. EDA and feature engineering.** Exploratory data analysis of the data, investigation of feature relation with the outcome and feature engineering using information from various sources.
- **Home credit default risk. Default prediction and hyperparameter tuning.** A range of different models on a subset of data is tried out to derive the most promising ones. The best performing ones are optimized further with hyperparameter tuning. Finally, the model is put in practical terms to evaluate the advantage of applying interest rate adjustments based on the predicted model probabilities.
- **Home credit default risk. External source 3 prediction and hyperparameter tuning.** A range of different models on a subset of data is tried out to derive the most promising ones. The best performing ones are optimized further with hyperparameter tuning.

Folders
- **helpers.** 2 .py files with helper functions for EDA and model building.
- **app.** contains the necessary files to deploy the loan outcome prediction model - Docker file, .yaml and app code itself. The model was deployed using the Google Cloud Platform, App Engine.
