import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import joblib
import lightgbm
import phik

from typing import List, Tuple, Dict, TypeVar, Type, Callable
from sklearn.base import RegressorMixin, TransformerMixin, ClassifierMixin

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split, cross_validate, \
                                    RandomizedSearchCV, StratifiedKFold, KFold

from sklearn.metrics import recall_score, average_precision_score, accuracy_score, \
                            precision_recall_curve, precision_score, r2_score, \
                            mean_squared_error, roc_auc_score, f1_score, \
                            classification_report, ConfusionMatrixDisplay, \
                            balanced_accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, \
                             GradientBoostingRegressor, RandomForestRegressor

from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn import set_config

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.linear_model import LinearRegression, SGDRegressor

from category_encoders.ordinal import OrdinalEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.helmert import HelmertEncoder
from category_encoders.count import CountEncoder
from category_encoders.one_hot import OneHotEncoder
from category_encoders.binary import BinaryEncoder

from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight

import pickle as pkl
from feature_engine.selection import DropConstantFeatures, DropCorrelatedFeatures
from feature_engine.imputation import RandomSampleImputer, AddMissingIndicator

set_config(transform_output = "pandas")
import optuna

import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Type definition
Predictor = TypeVar('Predictor')
Encoder = TypeVar('Encoder')
Trial = TypeVar('Trial')

def make_preprocessing_pipeline(X: pd.DataFrame, 
                                numeric: List[str], 
                                categorical: List[str],
                                cat_encoder: Encoder = 'ordinal',
                                imputer_strategy: str = None
                                ) -> Pipeline:
  '''Make preprocessing pipeline using the defined input functions'''
  
  if cat_encoder == 'one_hot':
      cat_encoder = OneHotEncoder()
  elif cat_encoder == 'ordinal':
      cat_encoder = OrdinalEncoder()
  elif cat_encoder == 'sum':
      cat_encoder = SumEncoder()
  elif cat_encoder == 'target':
      cat_encoder = TargetEncoder()
  elif cat_encoder == 'mestimate':
      cat_encoder = MEstimateEncoder()
  elif cat_encoder == 'helmer':
      cat_encoder = HelmertEncoder()
  elif cat_encoder == 'count':
      cat_encoder = CountEncoder()
  elif cat_encoder == 'binary':
      cat_encoder =  BinaryEncoder()      
  
  if imputer_strategy == 'constant':
      imputer_num = SimpleImputer(strategy='constant')
      imputer_cat = SimpleImputer(strategy='constant', fill_value='NaN')
      
  else:
      imputer_num = SimpleImputer(strategy='median')
      imputer_cat = SimpleImputer(strategy='most_frequent')

  numeric_transformer = Pipeline([
      ('imputer', imputer_num),
      ('std_scaler', StandardScaler())
      ])

  categorical_transformer = Pipeline([
      ('imputer', imputer_cat),
      ('cat_encoder', cat_encoder)
      ])

  preprocessing_pipeline = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric),
            ("categorical", categorical_transformer, categorical)
            ], 
        remainder="passthrough"
        )

  return preprocessing_pipeline

def fit_evaluate_classifiers(preprocessing_pipeline: Pipeline, 
                             classifiers: List[ClassifierMixin], 
                             X: pd.DataFrame, 
                             y: pd.DataFrame) -> Dict:
  '''Fit preprocessing and model training pipeline to the data and get 
  cross-validation metrics for provided scoring list'''
  
  scoring = ['recall', 'precision', 'f1', 'balanced_accuracy', 'average_precision', 'roc_auc']
  scores_dict = {'precision':[], 'recall':[], 'f1':[], 'balanced_accuracy':[], 'pr_auc':[], 'roc_auc':[]}

  for classifier in classifiers:
    steps = [
        ('preprocess', preprocessing_pipeline),
        ('clf', classifier)
        ]

    pipeline = Pipeline(steps)

    cv = StratifiedKFold(n_splits=5)
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=cv, n_jobs=-1)

    scores_dict['precision'].append(round(np.mean(scores['test_precision']), 4))
    scores_dict['recall'].append(round(np.mean(scores['test_recall']), 4))
    scores_dict['f1'].append(round(np.mean(scores['test_f1']), 4))
    scores_dict['balanced_accuracy'].append(round(np.mean(scores['test_balanced_accuracy']), 4))
    scores_dict['pr_auc'].append(round(np.mean(scores['test_average_precision']), 4))
    scores_dict['roc_auc'].append(round(np.mean(scores['test_roc_auc']), 4))

  return scores_dict

def fit_evaluate_preprocessors(preprocessor: Pipeline, 
                               classifiers: List[ClassifierMixin], 
                               X: pd.DataFrame, 
                               y: pd.DataFrame) -> Dict:
  '''Fit preprocessing and model training pipeline to the data and get 
  cross-validation metrics for provided scoring list'''
  
  scoring = ['recall', 'precision', 'f1', 'balanced_accuracy', 'average_precision', 'roc_auc']
  scores_dict = {'precision':[], 'recall':[], 'f1':[], 'balanced_accuracy':[], 'pr_auc':[], 'roc_auc':[]}
  
  for classifier in classifiers:
      
      steps = [
            ('preprocess', preprocessor),
            ('constant', DropConstantFeatures(tol=0.99)),
            ('correlated', DropCorrelatedFeatures(variables=None, 
                                      method='pearson', 
                                      threshold=0.9)),
            ('clf', classifier)
            ]
  
      pipeline = Pipeline(steps)
    
      cv = StratifiedKFold(n_splits=5)
      scores = cross_validate(pipeline, X, y, scoring=scoring, cv=cv, n_jobs=-1)
    
      scores_dict['precision'].append(round(np.mean(scores['test_precision']), 4))
      scores_dict['recall'].append(round(np.mean(scores['test_recall']), 4))
      scores_dict['f1'].append(round(np.mean(scores['test_f1']), 4))
      scores_dict['balanced_accuracy'].append(round(np.mean(scores['test_balanced_accuracy']), 4))
      scores_dict['pr_auc'].append(round(np.mean(scores['test_average_precision']), 4))
      scores_dict['roc_auc'].append(round(np.mean(scores['test_roc_auc']), 4))

  return scores_dict

def fit_evaluate_classifier(preprocessor: Pipeline, 
                               classifier: List[ClassifierMixin], 
                               X: pd.DataFrame, 
                               y: pd.DataFrame) -> Dict:
  '''Fit preprocessing and model training pipeline to the data and get 
  cross-validation metrics for provided scoring list'''
  
  scoring = ['recall', 'precision', 'f1', 'balanced_accuracy', 'average_precision', 'roc_auc']
  scores_dict = {'precision':[], 'recall':[], 'f1':[], 'balanced_accuracy':[], 'pr_auc':[], 'roc_auc':[]}

  steps = [
        ('preprocess', preprocessor),
        ('clf', classifier)
        ]
  
  pipeline = Pipeline(steps)

  cv = StratifiedKFold(n_splits=5)
  scores = cross_validate(pipeline, X, y, scoring=scoring, cv=cv, n_jobs=-1)

  scores_dict['precision'].append(round(np.mean(scores['test_precision']), 4))
  scores_dict['recall'].append(round(np.mean(scores['test_recall']), 4))
  scores_dict['f1'].append(round(np.mean(scores['test_f1']), 4))
  scores_dict['balanced_accuracy'].append(round(np.mean(scores['test_balanced_accuracy']), 4))
  scores_dict['pr_auc'].append(round(np.mean(scores['test_average_precision']), 4))
  scores_dict['roc_auc'].append(round(np.mean(scores['test_roc_auc']), 4))

  return scores_dict

def plot_performance(scores: Dict, 
                     label: List[str], 
                     c_min: int=0, 
                     c_max: int=1,
                     fig_size: tuple = (9, 6)):
  '''Plot provided performance metrics. Metrics are plotted as rows and model
  labels as columns'''
  
  scoring = [*scores]

  temp = []
  for key in scores:
    temp.append(scores[key])
  
  performance = np.vstack(temp)

  plt.figure(figsize=fig_size)
  cmap = sns.color_palette("ch:start=.1, rot=-.3", as_cmap=True)
  sns.heatmap(performance.T, xticklabels=scoring, yticklabels=label, 
              vmin=c_min, vmax=c_max, annot=True, fmt='g', cmap=cmap)
  plt.xticks(rotation=0)
  
def train_val_test_split(X: pd.DataFrame, 
                         y: pd.DataFrame , 
                         test_size: float, 
                         val_size: float
                         ) -> List[pd.DataFrame]:
    '''Split data into training, validation and testing sets based on given
    test and validation sizes'''
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size + val_size, 
                                                        stratify=y,
                                                        random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                     test_size=test_size, 
                                                     stratify=y_test,
                                                     random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test 

def select_subset(X: pd.DataFrame, 
                  y: pd.DataFrame, 
                  proportion: float,
                 ) -> List[pd.DataFrame]:
    '''Take a full dataset and return a stratified subset of it.'''
    
    X_train, X_subset, y_train, y_subset = train_test_split(X, y,
                                                        test_size=proportion, 
                                                        stratify=y,
                                                        random_state=45)
    
    return X_subset, y_subset

def objective_gradboost(trial: Trial,
                        X: pd.DataFrame, 
                        y: pd.DataFrame, 
                        scoring: str) -> float:
  '''An objective function to tune hyperparameters of Gradient Boosting Classifier.
  Returns mean test score'''

  params = {
    "n_estimators": trial.suggest_int("n_estimators", 100, 5000, step = 100),
    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log = True),
    "max_depth": trial.suggest_int("max_depth", 3, 9),
    "subsample": trial.suggest_float("subsample", 0.5, 0.9, step = 0.1),
    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    "random_state": 42,
    }

  gb_class = GradientBoostingClassifier(**params)

  # Compute scores
  cv = StratifiedKFold(n_splits=5)
  scores = cross_validate(gb_class, X, y, cv=cv, scoring=scoring, n_jobs=-1, verbose=0)
  score = scores["test_score"].mean()

  return score

def objective_svc(trial: Trial,
                    X: pd.DataFrame, 
                    y: pd.DataFrame, 
                    scoring: str) -> float:
    '''An objective function to tune hyperparameters of Support Vector Machine 
    Classifier. Returns mean test score'''

    params = {
      "C": trial.suggest_float("C", 1, 1000, log=True),
      "degree": trial.suggest_int("degree", 1, 9, step=1),
      "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
      "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
      "probability": True,
      }

    svc_class = SVC(**params)

    # Compute scores
    cv = StratifiedKFold(n_splits=5)
    scores = cross_validate(svc_class, X, y, cv=cv, scoring=scoring, n_jobs=-1, verbose=0)
    score = scores["test_score"].mean()

    return score
    

def run_optuna_study(X: pd.DataFrame, 
                     y: pd.DataFrame, 
                     objective: Callable, 
                     direction: str, 
                     scoring: str, 
                     n_trials: int=100):
    '''Function for enabling optuna study'''
    
    
    func = lambda trial: objective(trial, X, y, scoring=scoring)
    study = optuna.create_study(direction=direction)
    study.optimize(func, n_trials=n_trials)
    
    return study.best_params

def evaluate_model(model: Predictor, 
                   y: np.array,
                   y_predicted: np.array
                   ) -> Dict:
    '''Plot model performance evaluations: confusion matrix and classification
    report'''
    
    scoring = ['recall', 'precision', 'f1', 'average_precision', 'roc_auc']
    scores_dict = {'precision':[], 'recall':[], 'f1':[],  'pr_auc':[], 'roc_auc':[]}
    target_labels = ['non-default', 'default']

    scores_dict['precision'].append(round(precision_score(y, y_predicted), 4))
    scores_dict['recall'].append(round(recall_score(y, y_predicted), 4))
    scores_dict['f1'].append(round(f1_score(y, y_predicted), 4))
    scores_dict['pr_auc'].append(round(average_precision_score(y, y_predicted), 4))
    scores_dict['roc_auc'].append(round(roc_auc_score(y, y_predicted), 4))
    
    ConfusionMatrixDisplay.from_predictions(y, y_predicted, 
                                            display_labels=target_labels)

    plt.title("Confusion Matrix: ")
    plt.show()
    print(classification_report(y, y_predicted, target_names=target_labels))

    return scores_dict

def objective_catboost(trial: Trial,
                        X: pd.DataFrame, 
                        y: pd.DataFrame, 
                        scoring: str) -> float:
    '''An objective function to tune hyperparameters of CatBoosting Classifier.
    Returns mean test score'''
    
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', 
                               classes=classes,
                               y=y)

    class_weights = dict(zip(classes, weights))

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 5000, step = 100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True), 
        "objective": "Logloss",
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 5.5),
        "class_weights": class_weights, 
        "verbose": 0,
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "used_ram_limit": "3gb",
    }

    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    cat_class = CatBoostClassifier(**params)

    cv = StratifiedKFold(n_splits=5)
    scores = cross_validate(cat_class, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    score = scores["test_score"].mean()
    
    return score

def objective_catboost_regression(trial: Trial,
                                X: pd.DataFrame, 
                                y: pd.DataFrame, 
                                scoring: str) -> float:
    '''An objective function to tune hyperparameters of CatBoost Regression.
    Returns mean test score'''

    params = {}
    params['learning_rate'] = trial.suggest_float("learning_rate", 1e-4, 0.3, log=True)
    params['depth'] = trial.suggest_int("depth", 1, 12)
    params['l2_leaf_reg'] = trial.suggest_float('l2_leaf_reg', 1.0, 5.5)
    params['min_child_samples'] = trial.suggest_categorical('min_child_samples', [1, 4, 8, 16, 32])
    params['grow_policy'] = 'Depthwise'
    params['iterations'] = 10000
    params['od_type'] = 'Iter'
    params['od_wait'] = 20
    params['random_state'] = 42
    params['logging_level'] = 'Silent'
    params["bootstrap_type"] = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])

    cat_reg = CatBoostRegressor(**params)

    cv = KFold(n_splits=5)
    scores = cross_validate(cat_reg, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    score = scores["test_score"].mean()
    
    return score

def objective_lightgbm(trial: Trial,
                        X: pd.DataFrame, 
                        y: pd.DataFrame, 
                        scoring: str) -> float:
    '''An objective function to tune hyperparameters of Lightgbm Classifier.
    Returns mean test accuracy'''

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 100, 5000, step = 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
        "is_unbalance": trial.suggest_categorical("is_unbalance", [True, False]),
        "verbose": 0
    }

    # Add a callback for pruning.
    lgbm_class = LGBMClassifier(**params)

    # Compute scores
    cv = StratifiedKFold(n_splits=5)
    scores = cross_validate(lgbm_class, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    score = scores["test_score"].mean()
    
    return score


def plot_importance_svc(model: Predictor,
                    feature_list: List[str],
                    n_features: int=70):
    
    '''Plot the feature importance scores derived from the provided trained model'''
    metric = pd.concat([pd.Series(feature_list), 
                        pd.Series(model.coef_.reshape(-1))],
                       axis=1)
    
    metric.columns = ['feature_name', 'importance']
    metric_sorted = metric.sort_values('importance', ascending=False)[:n_features]
    
    plt.figure(figsize =(10,20))
    sns.barplot(x="importance", y="feature_name", data=metric_sorted)
    
    plt.ylabel('feature name')
    plt.xlabel('Coefficient names')
    plt.title('Model feature importances')
    
    return metric_sorted.feature_name.values


def plot_importance(model: Predictor,
                    feature_list: List[str],
                    n_features: int=70):
    '''Plot the feature importance scores derived from the provided trained model'''
    
    metric = pd.concat([pd.Series(feature_list), 
                        pd.Series(model.feature_importances_)],
                        axis=1)
    
    metric.columns = ['feature_name', 'importance']
    metric_sorted = metric.sort_values('importance', ascending=False)[:n_features]
    
    plt.figure(figsize =(10,20))
    sns.barplot(x="importance", y="feature_name", data=metric_sorted)
    
    plt.ylabel('feature name')
    plt.xlabel('importance score')
    plt.title('Model feature importances')
    
    return metric_sorted.feature_name.values

def fit_evaluate_regressors(preprocessing_pipeline: Pipeline, 
                            regressors: List[RegressorMixin],
                            X: pd.DataFrame, 
                            y: pd.DataFrame) -> Dict:
  '''With specified pipeline and regressors, fit and cross-validate models. 
  Return results in a dictionary'''
  
  scoring = ['r2', 'neg_root_mean_squared_error']
  scores_dict = {'r2':[], 'RMSE':[]}

  for regressor in regressors:
    steps = [
            ('preprocess', preprocessing_pipeline),
            ('reg', regressor)
            ]

    pipeline = Pipeline(steps)

    cv = KFold(n_splits=5)
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=cv)

    scores_dict['r2'].append(round(np.mean(scores['test_r2']), 3))
    scores_dict['RMSE'].append(round(np.mean(scores['test_neg_root_mean_squared_error']), 3))

  return scores_dict