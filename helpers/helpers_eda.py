import numpy as np
import pandas as pd
import phik

import seaborn as sns
import matplotlib.pyplot as plt
import math

from typing import List, Tuple, Dict, TypeVar
from sklearn.base import RegressorMixin, TransformerMixin, ClassifierMixin
import pickle as pkl

def print_shape(df: pd.DataFrame) -> pd.DataFrame:
    '''Print original dataframe shape'''
    
    print(f'Original dataframe shape {df.shape}')
    return df

def copy_data(df: pd.DataFrame) -> pd.DataFrame:
    '''Create a copy of the original dataframe'''
    
    return df.copy()

def lower_column_names(df: pd.DataFrame) -> pd.DataFrame:
    '''Covert column names to lower case'''
    
    df.columns = df.columns.str.lower()
    return df

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    '''Check missing values and output column names if there are any'''
    
    if df.isnull().any().any():
        missing_values = df.isnull().any()
        print(f'No missing values in {list(df.columns[~missing_values])}')
        missing_percentage = df.isnull().mean()*100
        missing_percentage[missing_percentage > 0].sort_values().plot.barh()

        plt.title('Missing values')
        plt.xlabel('Proportion of data missing (%)')
        plt.ylabel('Variable name')
        plt.xlim(0, 100);
    else:
        print('No missing values')
    return df

def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    '''Remove duplicate rows from the dataset'''
    
    df.drop_duplicates(inplace=True)
    return df

def remove_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    '''Remove empty rows excluding the identification columns'''
    
    id_columns = df.filter(like='sk_id').columns
    nan_mask = df.drop(id_columns, axis=1).isna().all(axis=1)
    df.drop(df[nan_mask].index, inplace=True)
    return df
    
def print_output_shape(df: pd.DataFrame) -> pd.DataFrame:
    '''Print original dataframe shape'''
    
    print(f'Dataframe shape after cleaning {df.shape}')
    return df

def plot_outcome_distribution(df: pd.DataFrame, 
                              target: str, 
                              labels: List[str]) -> None:
    '''Plot features distribution divided into groups by outcome'''
    
    fig, ax = plt.subplots(1,2,figsize=(9, 4.5))
    plt.suptitle('Loan outcome')

    sns.countplot(data=df, x=target, ax=ax[0])
    ax[0].set_title('Value counts')
    ax[0].set_xlabel('')
    ax[0].set_ylabel('')
    ax[0].set_xticklabels(labels)

    df[target].value_counts().plot.pie(autopct='%.1f%%', 
                                       ax=ax[1], 
                                       shadow=True, 
                                       labels=labels)
    ax[1].set_title('Percentage')
    ax[1].set_ylabel('')

def top_numeric_correlations(df: pd.DataFrame, 
                             output: str, 
                             features: List[str], 
                             n_correlations: int
                            ) -> pd.DataFrame:
    '''Derive and plot the top n correlations between numeric features and
    output variable'''
    
    correlation_sorted = abs(df[features + [output]] \
                             .corr()[[output]]) \
                             .sort_values(by=output, 
                             ascending=False)[1:n_correlations+1]
    
    return correlation_sorted.style.background_gradient(vmin=0, vmax=1)

def plot_boxplots(data: pd.DataFrame, 
                  feature: str, 
                  categories: List[str], 
                  rows: int) -> None:
  '''Plot boxplots by subgroups of variables in a multi-plot'''

  n_figs = len(categories)
  fig, axes = plt.subplots(rows, math.ceil(n_figs/rows), figsize=(23, 10))
  fig.suptitle('Numeric value distribution in successful and default loan groups', fontsize=16) 
  axes = axes.flatten()

  empty_figs = len(axes) - n_figs
  if empty_figs >= 1:
    for fig_index in range(empty_figs):
        index = (len(axes)-fig_index) - 1
        axes[index].axis("off")
    
  for category, axis in zip(categories, axes):
    sns.boxplot(data=data, 
                y=category,
                x=feature, 
                ax=axis)
    
    axis.title.set_text(f'{category}')
    axis.set_xlabel('Loan')
    axis.set_xticklabels(['successful', 'default'])
    axis.set_ylabel(' ')
    
def top_categorical_correlations(df: pd.DataFrame,
                                 output: str, 
                                 n_correlations: int
                                 ) -> pd.DataFrame:
    '''Derive and plot the top n correlations between categorical features and
    output variable'''
    
    correlation = df.phik_matrix(dropna=False)
    correlation_sorted = correlation[['target']]\
                         .sort_values(by ='target',
                                      ascending=False)[1:n_correlations+1]
    
    return correlation_sorted.style.background_gradient(vmin=0, vmax=1)

def plot_density(data:pd.DataFrame, 
                  feature: str, 
                  categories: List[str], 
                  rows: int) -> None:
  '''Plot densities by subgroups of variables in a multi-plot'''

  n_figs = len(categories)
  fig, axes = plt.subplots(rows, math.ceil(n_figs/rows), figsize=(18, 7))
  fig.suptitle('Kernel density estimate in successful and default loan groups', fontsize=16) 
  axes = axes.flatten()
    
  empty_figs = len(axes) - n_figs
  if empty_figs >= 1:
    for fig_index in range(empty_figs):
        index = (len(axes)-fig_index) - 1
        axes[index].axis("off")

  for category, axis in zip(categories, axes):
    sns.kdeplot(data=data, 
                x=category, 
                hue=feature, 
                common_norm=False, 
                ax=axis)
    
    axis.legend(labels=['default', 'successful'])
    axis.set_ylabel(' ') 
    
def calculate_feature_aggregates(df: pd.DataFrame,
                                 columns: List[str], 
                                 calculations: List[str], 
                                 group_id: str, 
                                 sort_id: str
                                ) -> pd.DataFrame:
    '''For the provided column list calculate aggregations for the provided
    list of calculations. The values are grouped and sorted based on group_id 
    and sort_id respectively'''
    
    columns = columns + [group_id] + [sort_id]
    df_sorted = df[columns].sort_values(sort_id, ascending=False).drop(sort_id, axis=1)
        
    df_aggregated = df_sorted.groupby(group_id).agg(calculations)
    df_aggregated.columns = ['_'.join(name) for name in df_aggregated.columns.to_flat_index()]
    df_aggregated.reset_index(inplace=True)
    return df_aggregated

def adjust_feature_lists(df: pd.DataFrame, 
                         categorical: List[str], 
                         remove_columns: set
                        ) -> List[str]:
    '''Remove given columns from feature list'''
    
    numeric = df.columns[~df.columns.isin(categorical)].tolist()
    categorical = [col for col in categorical if col not in remove_columns]
    
    return categorical, numeric

def calculate_time_feature_aggregates(df: pd.DataFrame, 
                                      columns: List[str], 
                                      group_id: str, 
                                      sort_id: str, 
                                      calculations: List[str],
                                      n: int, 
                                      name=None) -> pd.DataFrame:
    '''For the provided column list calculate aggregations across n elements for
    the provided list of calculations. The values are grouped and sorted based 
    on group_id and sort_id respectively'''
    
    columns = columns + [group_id]
    df_sorted = df[columns].sort_values(sort_id, ascending=False).groupby(group_id).head(n).drop(sort_id, axis=1)
    df_aggregated_first = df_sorted.groupby(group_id).agg(calculations)
    df_aggregated_first = df_aggregated_first.add_suffix('_' + name + '_first' + str(n))

    df_sorted = df[columns].sort_values(sort_id, ascending=False).groupby(group_id).tail(n).drop(sort_id, axis=1)
    df_aggregated_last = df_sorted.groupby(group_id).agg(calculations)
    df_aggregated_last = df_aggregated_last.add_suffix('_' + name + '_last' + str(n))
    
    df_aggregated = df_aggregated_first.merge(df_aggregated_last, on=group_id, how='left')
    df_aggregated.columns = df_aggregated.columns.map('_'.join)
    
    return df_aggregated

def calculate_2feature_combinations(df: pd.DataFrame, 
                                    columns: List[str], 
                                    calculations: List[str]
                                   ) -> pd.DataFrame:
    '''Perfrom calulations between two columns from columns list'''
    
    for calculation in calculations:
        name = '_'.join(columns) + '_' + calculation
        if calculation == "ratio":
            df[name] = df[columns[0]]/ (df[columns[1]] + 0.0000001)
        if calculation == "diff":
            df[name] = df[columns[0]]/ (df[columns[1]] + 0.0000001)
        print(f'Created feature {name}')
    return df

def calculate_multi_feature_combinations(df: pd.DataFrame, 
                                         columns: List[str], 
                                         calculations: List[str]
                                         ) -> pd.DataFrame:
     '''Perfrom calculations between multiple columns from columns list'''
     for calculation in calculations:
        name = columns[0][:-2] + '_' + 'cols' + '_' + calculation
        if calculation == "mean":
            df[name] = df[columns].mean(axis=1)
        if calculation == "max":
            df[name] = df[columns].max(axis=1)
        if calculation == "min":
            df[name] = df[columns].min(axis=1)
        if calculation == "sum":
            df[name] = df[columns].sum()
        if calculation == "multi":
            df[name] = df[columns[0]]
            for column in columns[1:]:
                df[name] = df[name] * df[column]
        if calculation == "var":
             df[name] = df[columns].var(axis=1)   
        
        print(f'Created feature {name}') 
        
     return df

def ploth_category_counts_percentage(df: pd.DataFrame,
                                    feature: str, 
                                    group_variable: str
                                    ) -> None:
    '''Plot horizontal categorical variable counts and percentage'''
    
    
    fig_height = (df[feature].nunique()/2.5)
    fig, axis = plt.subplots(2,1,figsize=(7, fig_height))
    fig.tight_layout(h_pad=2)
    
    df[feature].value_counts(ascending=True) \
               .plot(kind = "barh", ax=axis[0], 
                     title = "Overall number of loans in each group")
    
    df_cross_tab = pd.crosstab(df[feature],
                 df[group_variable], 
                 normalize="index").sort_values(1) * 100
    
    df_cross_tab.plot(kind='barh', 
                    stacked=True, 
                     ax=axis[1], title = "Loan outcomes for each group")

    axis[1].legend(loc="lower left", ncol=2)
    plt.ylabel('')
    plt.xlabel("Proportion (%)")

    plt.legend(labels=['successful', 'default'], loc='lower left')

def plotv_category_counts_percentage(df: pd.DataFrame,
                                     feature: str, 
                                     group_variable: str):
    '''Plot vertical categorical variable counts and percentage'''
    fig, axis = plt.subplots(1,2,figsize=(6, 3))
    fig.tight_layout(w_pad=5)
    
    df[feature].value_counts(ascending=True) \
               .plot(kind = "bar", ax=axis[0], 
                     title = "Overall number of loans in each group")
    
    df_cross_tab = pd.crosstab(df[feature],
                 df[group_variable], 
                 normalize="index").sort_values(1, ascending=False) * 100
    
    df_cross_tab.plot(kind='bar', 
                     stacked=True, 
                     ax=axis[1], title = "Loan outcomes for each group")

    axis[1].legend(loc="lower left", ncol=2)
    plt.ylabel('')
    plt.xlabel("Proportion (%)")

    plt.legend(labels=['successful', 'default'], loc='lower left')
