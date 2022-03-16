import warnings
warnings.filterwarnings("ignore")

# Import libraries
import time as tm
import numpy as np
import pickle as pk
import pandas as pd
import random as rd
#---------------------------------------------------------------------------------------------------#
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule
#---------------------------------------------------------------------------------------------------#
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
#---------------------------------------------------------------------------------------------------#
from scipy.stats import pearsonr, chisquare, shapiro, spearmanr, mannwhitneyu
#---------------------------------------------------------------------------------------------------#
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, LabelBinarizer, MultiLabelBinarizer
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.metrics import accuracy_score, r2_score
#---------------------------------------------------------------------------------------------------#
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.linear_model import RidgeClassifier, Ridge
#---------------------------------------------------------------------------------------------------#
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
#---------------------------------------------------------------------------------------------------#
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
#---------------------------------------------------------------------------------------------------#
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
#---------------------------------------------------------------------------------------------------#
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#---------------------------------------------------------------------------------------------------#

import collections
class OrderedSet(collections.Set):
    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, element):
        return element in self.d

    def __iter__(self):
        return iter(self.d)

def load_models(args):
  # Initialize the model class
  njobs = args["njobs"]
  rands = args["rands"]
  models = [
      {
        #ALL DOCUMENDATION FEATURES
        "id": 0,
        "type": "classification",
        "model": RandomForestClassifier(),
        "principal": "RandomForest",
        "family": "ensemble",
        "search_space":
        [
          {
            "n_estimators": Integer(100, 500),
            "criterion": Categorical(['gini', 'entropy']),
            "max_depth": Integer(6, 20),
            "min_samples_split": Integer(2, 10),
            "min_samples_leaf": Integer(2, 10),
            "min_weight_fraction_leaf": Real(0, 0.5, prior='uniform'),
            "max_features": Categorical(['auto', 'sqrt','log2']),
            #"max_leaf_nodes": Integer(0, 10),
            #"min_impurity_decrease": Real(0, 1, prior='uniform'),
            "bootstrap": Categorical([False]),
            "oob_score": Categorical([False]),
            "n_jobs": [njobs],
            "random_state": [rands],
            #"verbose": Integer(0, 2),
            #"warm_start": Categorical([False, True]),
            "class_weight": Categorical(['balanced', 'balanced_subsample', None]),
            "ccp_alpha": Real(0, 1, prior='uniform'),
            #"max_samples": Integer(0, len(X)),
          },
          {
            "n_estimators": Integer(100, 500),
            "criterion": Categorical(['gini', 'entropy']),
            "max_depth": Integer(6, 20),
            "min_samples_split": Integer(2, 10),
            "min_samples_leaf": Integer(2, 10),
            "min_weight_fraction_leaf": Real(0, 0.5, prior='uniform'),
            "max_features": Categorical(['auto', 'sqrt','log2']),
            #"max_leaf_nodes": Integer(0, 10),
            #"min_impurity_decrease": Real(0, 1, prior='uniform'),
            "bootstrap": Categorical([True]),
            "oob_score": Categorical([False, True]),
            "n_jobs": [njobs],
            "random_state": [rands],
            #"verbose": Integer(0, 2),
            #"warm_start": Categorical([False, True])
            "class_weight": Categorical(['balanced', 'balanced_subsample', None]),
            "ccp_alpha": Real(0, 1, prior='uniform'),
            #"max_samples": Integer(0, len(X)),
          },
        ]
      },
      {
        "id": 1,
        "type": "classification",
        "model": DecisionTreeClassifier(),
        "principal": "DecisionTree",
        "family": "tree",
        "search_space":
        {
          "criterion": Categorical(['gini', 'entropy']),
          #"splitter": Categorical(['best', 'random']),
          #"max_depth": Integer(6, 20),
          "min_samples_split": Integer(2, 10),
          "min_samples_leaf": Integer(1, 10),
          "min_weight_fraction_leaf": Real(0, 0.5, prior='uniform'),
          "max_features": Categorical(['auto', 'sqrt','log2']),
          "random_state": [rands],
          #"max_leaf_nodes":Integer(0, 10),
          #"min_impurity_decrease": Real(0, 1, prior='uniform'),
          #"class_weight": Categorical(['dict', 'list', 'balanced']),
          #"alpha": Real(0, 1, prior='uniform'),
        }
      },
      {
        "id": 2,
        "type": "classification",
        "model": KNeighborsClassifier(),
        "principal": "KNeighbors",
        "family": "neighbors",
        "search_space":
        {
          "n_neighbors": Integer(1, 6),
          "weights": Categorical(['uniform', 'distance']),
          "algorithm": Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),
          "leaf_size": Integer(1, 50),
          "p": Integer(1, 2),
          "metric": Categorical(['minkowski']),
          #"metric_params": Categorical(['']),
          "n_jobs": [njobs],
        }
      },
      {
        "id": 3,
        "type": "classification",
        "model": SVC(),
        "principal": "SupportVectors",
        "family": "svm",
        "search_space":
        [
         {
          "C": Integer(1, 10),
          "kernel": Categorical(['linear']), #precomputed #Precomputed matrix must be a square matrix
          #"shrinking": Categorical([True, False]),
          #"probability": Categorical([True, False]),
          "tol": [0.00001],
          #"cache_size": Integer(100, 10000),
          #"class_weight": Categorical(['dict', 'balanced']),
          #"verbose": Categorical([True, False]),
          "max_iter": [1000],
          #"shrinking": Categorical(['ovo', 'ovr']),
          #"break_ties": Categorical([True, False]),
          "random_state": [rands],
        },
        #{
        #  "C": Integer(1, 10),
        #  "kernel": Categorical(['rbf']), #precomputed #Precomputed matrix must be a square matrix
        #  "gamma": Categorical(['scale', 'auto']),
        #  #"shrinking": Categorical([True, False]),
        #  #"probability": Categorical([True, False]),
        #  "tol": [0.00001],
        #  #"cache_size": Integer(100, 10000),
        #  #"class_weight": Categorical(['dict', 'balanced']),
        #  #"verbose": Categorical([True, False]),
        #  "max_iter": [1000],
        #  #"shrinking": Categorical(['ovo', 'ovr']),
        #  #"break_ties": Categorical([True, False]),
        #  "random_state": [rands],
        #},
        #{
        #  "C": Integer(1, 10),
        #  "kernel": Categorical(['sigmoid']), #precomputed #Precomputed matrix must be a square matrix
        #  "gamma": Categorical(['scale', 'auto']),
        #  "coef0": Real(0, 1, prior='uniform'),
        #  #"shrinking": Categorical([True, False]),
        #  #"probability": Categorical([True, False]),
        #  "tol": [0.00001],
        #  #"cache_size": Integer(100, 10000),
        #  #"class_weight": Categorical(['dict', 'balanced']),
        #  #"verbose": Categorical([True, False]),
        #  "max_iter": [1000],
        #  #"shrinking": Categorical(['ovo', 'ovr']),
        #  #"break_ties": Categorical([True, False]),
        #  "random_state": [rands],
        #},
        {
          "C": Integer(1, 10),
          "kernel": Categorical(['poly']), #precomputed #Precomputed matrix must be a square matrix
          "degree": [1,2,3],
          "gamma": Categorical(['scale', 'auto']),
          "coef0": Real(0, 1, prior='uniform'),
          #"shrinking": Categorical([True, False]),
          #"probability": Categorical([True, False]),
          "tol": [0.00001],
          #"cache_size": Integer(100, 10000),
          #"class_weight": Categorical(['dict', 'balanced']),
          #"verbose": Categorical([True, False]),
          "max_iter": [1000],
          #"shrinking": Categorical(['ovo', 'ovr']),
          #"break_ties": Categorical([True, False]),
          "random_state": [rands],
        }
       ]
      },
      {
        "id": 4,
        "type": "classification",
        "model": SGDClassifier(),
        "principal": "StohasticGradient",
        "family": "linear",
        "search_space":
        [
         {
          "loss": Categorical(['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']),
          "penalty": Categorical(['l2', 'l1']),
          "alpha": Real(0, 0.5, prior='uniform'),
          "fit_intercept": Categorical([True, False]),
          #"max_iter": Integer(500, 1000),
          #"tol": Real(0, 0.5, prior='uniform'),
          #"shuffle": Categorical([True, False]),
          #"verbose": Categorical([True, False]),
          #"epsilon": Real(0, 0.5, prior='uniform'),
          "n_jobs": [njobs],
          "random_state": [rands],
          "learning_rate": Categorical(['constant', 'optimal', 'invscaling', 'adaptive']),
          "eta0": Real(0, 0.5, prior='uniform'), #eta0 must be > 0
          #"power_t": Real(0, 0.5, prior='uniform'),
          #"early_stopping": Categorical([True, False]),
          #"validation_fraction": Real(0, 0.5, prior='uniform'),
          #"n_iter_no_change": Integer(1, 10),
          #"class_weight": Categorical(['dict', 'balanced', 'None']),
          #"warm_start": Categorical([True, False]),
          #"average": Categorical([True, False]),
         },
         {
          "loss": Categorical(['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']),
          "penalty": Categorical(['elasticnet']),
          "alpha": Real(0, 0.5, prior='uniform'),
          "l1_ratio": Real(0, 0.5, prior='uniform'),
          "fit_intercept": Categorical([True, False]),
          #"max_iter": Integer(500, 1000),
          #"tol": Real(0, 0.5, prior='uniform'),
          #"shuffle": Categorical([True, False]),
          #"verbose": Categorical([True, False]),
          #"epsilon": Real(0, 0.5, prior='uniform'),
          "n_jobs": [njobs],
          "random_state": [rands],
          "learning_rate": Categorical(['constant', 'optimal', 'invscaling', 'adaptive']),
          "eta0": Real(0, 0.5, prior='uniform'), #eta0 must be > 0
          #"power_t": Real(0, 0.5, prior='uniform'),
          #"early_stopping": Categorical([True, False]),
          #"validation_fraction": Real(0, 0.5, prior='uniform'),
          #"n_iter_no_change": Integer(1, 10),
          #"class_weight": Categorical(['dict', 'balanced', 'None']),
          #"warm_start": Categorical([True, False]),
          #"average": Categorical([True, False]),
        }
       ]
      },
      {
        "id": 5,
        "type": "classification",
        "model": RidgeClassifier(),
        "principal": "Ridge",
        "family": "linear",
        "search_space":
        [
          {
          "alpha": Real(0, 0.5, prior='uniform'),
          "fit_intercept": Categorical([True, False]),
          "normalize": Categorical([True, False]),
          "copy_X": Categorical([True, False]),
          #"max_iter": Integer(500, 1000),
          #"tol": Real(0, 0.5, prior='uniform'),
          #"class_weight": Categorical(['dict', 'balanced', 'None']),
          "solver": Categorical(['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
          "positive": Categorical([False]),
          "random_state": [rands],
        },
        {
          "alpha": Real(0, 0.5, prior='uniform'),
          "fit_intercept": Categorical([True, False]),
          "normalize": Categorical([True, False]),
          "copy_X": Categorical([True, False]),
          #"max_iter": Integer(500, 1000),
          #"tol": Real(0, 0.5, prior='uniform'),
          #"class_weight": Categorical(['dict', 'balanced', 'None']),
          "solver": Categorical(['lbfgs']),
          "positive": Categorical([True]),
          "random_state": [rands],
        }
       ]
      },

      {
        "id": 6,
        "type": "classification",
        "model": LogisticRegression(),
        "search_space":
        [
         {
          "penalty": Categorical(['l1']),
          "C": Integer(1, 2), #Real(0, 1, prior='uniform')
          #"tol": Real(0, 0.5, prior='uniform'),
          #"max_iter": Integer(500, 1000), #the higher the more time
          "fit_intercept": Categorical([True, False]),
          #"intercept_scaling": Real(0.5, ``.5, prior='uniform'),
          "class_weight": Categorical(['balanced', None]),
          "random_state": [rands],
          "solver": Categorical(['liblinear']),
          #"verbose": Integer(0, 2),
          "n_jobs": [njobs],
         },
         {
          "penalty": Categorical(['l2']),
          "C": Integer(1, 2), #Real(0, 1, prior='uniform')
          #"tol": Real(0, 0.5, prior='uniform'),
          "dual": Categorical([True, False]),
          "fit_intercept": Categorical([True, False]),
          "class_weight": Categorical(['balanced', None]),
          "random_state": [rands],
          "solver": Categorical(['liblinear']),
          #"verbose": Integer(0, 2),
          "n_jobs": [njobs],
         },
         {
          "penalty": Categorical(['l2', 'none']),
          "C": Integer(1, 2), #Real(0, 1, prior='uniform')
          #"tol": Real(0, 0.5, prior='uniform'),
          "fit_intercept": Categorical([True, False]),
          "class_weight": Categorical(['balanced', None]),
          "random_state": Integer(0, 10),
          "solver": Categorical(['lbfgs', 'newton-cg', 'sag']),
          #"verbose": Integer(0, 2),
          #"warm_start": Categorical([True, False]),
          "n_jobs": [njobs],
         },
         {
          "penalty": Categorical(['l1', 'l2', 'none']),
          "C": Integer(1, 2), #Real(0, 1, prior='uniform')
          #"tol": Real(0, 0.5, prior='uniform'),
          "fit_intercept": Categorical([True, False]),
          "class_weight": Categorical(['balanced', None]),
          "random_state": [rands],
          "solver": Categorical(['saga']),
          #"verbose": Integer(0, 2),
          #"warm_start": Categorical([True, False]),
          "n_jobs": [njobs],
         },
         {
          "penalty": Categorical(['elasticnet']),
          "C": Integer(1, 2), #Real(0, 1, prior='uniform')
          #"tol": Real(0, 0.5, prior='uniform'),
          "fit_intercept": Categorical([True, False]),
          "class_weight": Categorical(['balanced', None]),
          "random_state": [rands],
          "solver": Categorical(['saga']),
          #"verbose": Integer(0, 2),
          #"warm_start": Categorical([True, False]),
          "n_jobs": [njobs],
          "l1_ratio": Real(0, 1, prior='uniform'),
         },
        ]
      },
      {
        "id": 7,
        "type": "classification",
        "model": Perceptron(),
        "search_space":
        [
         {
          "penalty": Categorical(['l2', 'l1']),
          "alpha": Real(0, 0.5, prior='uniform'),
          "fit_intercept": Categorical([True, False]),
          #"max_iter": Integer(500, 1000),
          #"tol": Real(0, 0.5, prior='uniform'),
          #"shuffle": Categorical([True, False]),
          #"verbose": Categorical([True, False]),
          "eta0": Real(0, 0.5, prior='uniform'), #eta0 must be > 0
          "n_jobs": [njobs],
          "random_state": [rands],
          #"early_stopping": Categorical([True, False]),
          #"validation_fraction": Real(0, 0.5, prior='uniform'),
          #"n_iter_no_change": Integer(1, 10),
          #"class_weight": Categorical(['dict', 'balanced', 'None']),
          #"warm_start": Categorical([True, False]),
         },
         {
          "penalty": Categorical(['elasticnet']),
          "alpha": Real(0, 0.5, prior='uniform'),
          "l1_ratio": Real(0, 0.5, prior='uniform'),
          "fit_intercept": Categorical([True, False]),
          #"max_iter": Integer(500, 1000),
          #"tol": Real(0, 0.5, prior='uniform'),
          #"shuffle": Categorical([True, False]),
          #"verbose": Categorical([True, False]),
          "eta0": Real(0, 0.5, prior='uniform'), #eta0 must be > 0
          "n_jobs": [njobs],
          "random_state": [rands],
          #"early_stopping": Categorical([True, False]),
          #"validation_fraction": Real(0, 0.5, prior='uniform'),
          #"n_iter_no_change": Integer(1, 10),
          #"class_weight": Categorical(['dict', 'balanced', 'None']),
          #"warm_start": Categorical([True, False]),
         }
        ]
      },
      {
        "id": 8,
        "type": "classification",
        "model": PassiveAggressiveClassifier(),
        "search_space":
        {
          "C": Real(0, 1, prior='uniform'), #eta0 must be > 0
          "fit_intercept": Categorical([True, False]),
          #"max_iter": Integer(500, 1000),
          #"tol": Real(0, 0.5, prior='uniform'),
          #"early_stopping": Categorical([True, False]),
          #"validation_fraction": Real(0, 0.5, prior='uniform'),
          #"n_iter_no_change": Integer(1, 10),
          #"shuffle": Categorical([True, False]),
          #"verbose": Categorical([True, False]),
          "loss": Categorical(['hinge', 'log', 'modified_huber', 'squared_hinge', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
          "n_jobs": [njobs],
          "random_state": [rands],
          #"warm_start": Categorical([True, False]),
          #"class_weight": Categorical(['dict', 'balanced', 'None']),
          #"average": Categorical([True, False]),
        }
      },
      {
        "id": 9,
        "type": "classification",
        "model": LinearDiscriminantAnalysis(),
        "search_space":
        {
          "solver": Categorical(['svd', 'lsqr']) #kd_tree #unknown solver kd_tree (valid solvers are 'svd', 'lsqr', and 'eigen').
        }
      },
      {
        "id": 10,
        "type": "classification",
        "model": MultinomialNB(),
        "search_space":
        {
          "fit_prior": Categorical([True, False]),
          "alpha": Real(0, 1, prior='uniform')
        }
      },
      {
        "id": 11,
        "type": "classification",
        "model": GaussianNB(),
        "search_space":
        {
          "var_smoothing": Real(0, 1, prior='uniform')
        }
      },
      {
        "id": 12,
        "type": "classification",
        "model": BernoulliNB(),
        "search_space":
        {
          "fit_prior": Categorical([True, False]),
          "alpha": Real(0, 1, prior='uniform')
        }
      },

      {
        #ALL DOCUMENDATION FEATURES
        "id": 0,
        "type": "regression",
        "model": RandomForestRegressor(),
        "principal": "RandomForest",
        "family": "ensemble",
        "search_space":
        [
          {
            "n_estimators": Integer(100, 500),
            "criterion": Categorical(['mse']), # 'squared_error', 'absolute_error', 'poisson'
            "max_depth": Integer(6, 20), # values of max_depth are integers from 6 to 20
            "min_samples_split": Integer(2, 10),
            "min_samples_leaf": Integer(1, 10),
            "min_weight_fraction_leaf": Real(0, 0.5, prior='uniform'),
            "max_features": Categorical(['auto', 'sqrt','log2']), 
            #"max_leaf_nodes": Integer(0, 10),
            #"min_impurity_decrease": Real(0, 1, prior='uniform'),
            "bootstrap": Categorical([False]), # values for boostrap can be either True or False
            "oob_score": Categorical([False]),
            "n_jobs": [njobs],
            "random_state": [rands],
            #"verbose": Integer(0, 2),
            #"warm_start": Categorical([False, True]),
            "ccp_alpha": Real(0, 1, prior='uniform'),
            #"max_samples": Integer(0, len(X)),
          },
          {
            "n_estimators": Integer(100, 500),
            "criterion": Categorical(['mse']), # 'squared_error', 'absolute_error', 'poisson'
            "max_depth": Integer(6, 20),
            "min_samples_split": Integer(2, 10),
            "min_samples_leaf": Integer(2, 10),
            "min_weight_fraction_leaf": Real(0, 0.5, prior='uniform'),
            "max_features": Categorical(['auto', 'sqrt','log2']),
            #"max_leaf_nodes": Integer(0, 10),
            #"min_impurity_decrease": Real(0, 1, prior='uniform'),
            "bootstrap": Categorical([True]),
            "oob_score": Categorical([False, True]),
            "n_jobs": [njobs],
            "random_state": [rands],
            #"verbose": Integer(0, 2),
            #"warm_start": Categorical([False, True])
            "ccp_alpha": Real(0, 1, prior='uniform'),
            #"max_samples": Integer(0, len(X)),
          }
        ]
      },
      {
        #ALL DOCUMENDATION FEATURES
        "id": 1,
        "type": "regression",
        "model": DecisionTreeRegressor(),
        "principal": "DecisionTree",
        "family": "tree",
        "search_space":
        {
          "criterion": Categorical(['mse', 'friedman_mse', 'mae']), # 'squared_error', 'absolute_error', 'poisson'
          #"splitter": Categorical(['best', 'random']),
          #"max_depth": Integer(6, 20), # values of max_depth are integers from 6 to 20
          "min_samples_split": Integer(2, 10),
          "min_samples_leaf": Integer(2, 10),
          "min_weight_fraction_leaf": Real(0, 0.5, prior='uniform'),
          "max_features": Categorical(['auto', 'sqrt','log2']),
          "random_state": [rands],
          #"max_leaf_nodes":Integer(0, 10),
          #"min_impurity_decrease": Real(0, 1, prior='uniform'),
          #"alpha": Real(0, 1, prior='uniform')
        }
      },
      {
        "id": 2,
        "type": "regression",
        "model": KNeighborsRegressor(),
        "search_space":
        {
          "n_neighbors": Integer(1, 10),
          "weights": Categorical(['uniform', 'distance']),
          "algorithm": Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']), 
          "leaf_size": Integer(1, 50),
          "p": Integer(1, 2),
          "metric": Categorical(['minkowski']),
          #metric_params": Categorical(['']),
          "n_jobs": [njobs],
        }
      },
      {
        "id": 3,
        "type": "regression",
        "model": SVR(),
        "search_space":
        [
         {
          "C": Real(0.1, 10, prior='uniform'),
          "kernel": Categorical(['linear']), #precomputed #Precomputed matrix must be a square matrix
          "gamma": Categorical(['scale']), #auto #will chose one of the existing
          #"shrinking": Categorical([True, False]),
          #"probability": Categorical([True, False]),
          "tol": [0.01],
          #"epsilon": Real(0, 0.5, prior='uniform'),
          #"shrinking": Categorical(['ovo', 'ovr']),
          #"cache_size": Integer(1, 500),
          #"verbose": Categorical([True, False]),
          #"max_iter": Integer(-1, 1000),
         },
         #{
         # "C": Real(0.1, 10, prior='uniform'),
         # "kernel": Categorical(['rbf']), #precomputed #Precomputed matrix must be a square matrix
         # "gamma": Categorical(['scale']), #auto #will chose one of the existing
         # #"shrinking": Categorical([True, False]),
         # #"probability": Categorical([True, False]),
         # "tol": [0.01],
         # #"epsilon": Real(0, 0.5, prior='uniform'),
         # #"shrinking": Categorical(['ovo', 'ovr']),
         # #"cache_size": Integer(1, 500),
         # #"verbose": Categorical([True, False]),
         # #"max_iter": Integer(-1, 1000),
         #},
         #{
         # "C": Real(0.1, 10, prior='uniform'),
         # "kernel": Categorical(['sigmoid']), #precomputed #Precomputed matrix must be a square matrix
         # "gamma": Categorical(['scale']), #auto #will chose one of the existing
         # "coef0": Real(0, 1, prior='uniform'),
         # #"shrinking": Categorical([True, False]),
         # #"probability": Categorical([True, False]),
         # "tol": [0.01],
         # #"epsilon": Real(0, 0.5, prior='uniform'),
         # #"shrinking": Categorical(['ovo', 'ovr']),
         # #"cache_size": Integer(1, 500),
         # #"verbose": Categorical([True, False]),
         # #"max_iter": Integer(-1, 1000),
         #},
         {
          "C": Real(0.1, 10, prior='uniform'),
          "kernel": Categorical(['poly']), #precomputed #Precomputed matrix must be a square matrix
          "degree": [1],
          "gamma": Categorical(['scale']), #auto #will chose one of the existing
          "coef0": Real(0, 1, prior='uniform'),
          #"shrinking": Categorical([True, False]),
          #"probability": Categorical([True, False]),
          "tol": [0.01],
          #"epsilon": Real(0, 0.5, prior='uniform'),
          #"shrinking": Categorical(['ovo', 'ovr']),
          #"cache_size": Integer(1, 500),
          #"verbose": Categorical([True, False]),
          #"max_iter": Integer(-1, 1000),
         }
        ]
      },
      {
        "id": 4,
        "type": "regression",
        "model": SGDRegressor(),
        "principal": "StohasticGradient",
        "family": "linear",
        "search_space":
        [
         {
          "loss": Categorical(['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
          "penalty": Categorical(['l2', 'l1']),
          "alpha": Real(0, 0.5, prior='uniform'),
          "fit_intercept": Categorical([True, False]),
          #"max_iter": Integer(500, 1000),
          #"tol": Real(0, 0.5, prior='uniform'),
          #"shuffle": Categorical([True, False]),
          #"verbose": Categorical([True, False]),
          #"epsilon": Real(0, 0.5, prior='uniform'),
          "random_state": [rands],
          "learning_rate": Categorical(['constant', 'optimal', 'invscaling', 'adaptive']),
          "eta0": Real(0, 0.5, prior='uniform'), #eta0 must be > 0
          #"power_t": Real(0, 0.5, prior='uniform'),
          #"early_stopping": Categorical([True, False]),
          #"validation_fraction": Real(0, 0.5, prior='uniform'),
          #"n_iter_no_change": Integer(1, 10),
          #"warm_start": Categorical([True, False]),
          #"average": Categorical([True, False]),
         },
         {
          "loss": Categorical(['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
          "penalty": Categorical(['elasticnet']),
          "alpha": Real(0, 0.5, prior='uniform'),
          "l1_ratio": Real(0, 0.5, prior='uniform'),
          "fit_intercept": Categorical([True, False]),
          #"max_iter": Integer(500, 1000),
          #"tol": Real(0, 0.5, prior='uniform'),
          #"shuffle": Categorical([True, False]),
          #"verbose": Categorical([True, False]),
          #"epsilon": Real(0, 0.5, prior='uniform'),
          "random_state": [rands],
          "learning_rate": Categorical(['constant', 'optimal', 'invscaling', 'adaptive']),
          "eta0": Real(0, 0.5, prior='uniform'), #eta0 must be > 0
          #"power_t": Real(0, 0.5, prior='uniform'),
          #"early_stopping": Categorical([True, False]),
          #"validation_fraction": Real(0, 0.5, prior='uniform'),
          #"n_iter_no_change": Integer(1, 10),
          #"warm_start": Categorical([True, False]),
          #"average": Categorical([True, False]),
        }
       ]
      },
      {
        "id": 5,
        "type": "regression",
        "model": Ridge(),
        "principal": "StohasticGradient",
        "family": "linear",
        "search_space":
        [
         {
          "alpha": Real(0, 0.5, prior='uniform'),
          "fit_intercept": Categorical([True, False]),
          "normalize": Categorical([True, False]),
          "copy_X": Categorical([True, False]),
          #"max_iter": Integer(500, 1000),
          #"tol": Real(0, 0.5, prior='uniform'),
          "solver": Categorical(['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
          "positive": Categorical([False]),
          "random_state": [rands],
        },
        {
          "alpha": Real(0, 0.5, prior='uniform'),
          "fit_intercept": Categorical([True, False]),
          "normalize": Categorical([True, False]),
          "copy_X": Categorical([True, False]),
          #"max_iter": Integer(500, 1000),
          #"tol": Real(0, 0.5, prior='uniform'),
          "solver": Categorical(['lbfgs']),
          "positive": Categorical([True]),
          "random_state": [rands],
        }
       ]
      },
      {
        "id": 6,
        "type": "regression",
        "model": LinearRegression(),
        "search_space":
        {
          "fit_intercept": Categorical([True, False]),
          "normalize": Categorical([True, False]),
          #"copy_X": Categorical([True, False]),
          "n_jobs": [njobs],
          #"positive": Categorical([False, True]),
        }
      },
      {
        "id": 7,
        "type": "regression",
        "model": BayesianRidge(),
        "search_space":
        {
          "n_iter": [300],
          "tol": Real(0, 0.5, prior='uniform'),
          "alpha_1": Real(0, 0.5, prior='uniform'),
          "alpha_2": Real(0, 0.5, prior='uniform'),
          "lambda_1": Real(0, 0.5, prior='uniform'),
          "lambda_2": Real(0, 0.5, prior='uniform'),
          "alpha_init": Real(0, 0.5, prior='uniform'),
          "compute_score": Categorical([True, False]),
          "fit_intercept": Categorical([True, False]),
          "normalize": Categorical([True, False]),
          #"copy_X": Categorical([True, False]),
          #"verbose": Categorical([True, False]),
        }
      },
      {
        "id": 8,
        "type": "regression",
        "model": ARDRegression(),
        "search_space":
        {
          "n_iter": [300],
          "tol": Real(0, 0.5, prior='uniform'),
          "alpha_1": Real(0, 0.5, prior='uniform'),
          "alpha_2": Real(0, 0.5, prior='uniform'),
          "lambda_1": Real(0, 0.5, prior='uniform'),
          "lambda_2": Real(0, 0.5, prior='uniform'),
          "compute_score": Categorical([True, False]),
          "threshold_lambda":Real(10000, 20000, prior='uniform'),
          "fit_intercept": Categorical([True, False]),
          "normalize": Categorical([True, False]),
          #"copy_X": Categorical([True, False]),
          #"verbose": Categorical([True, False]),
        }
      }
  ]
  return models

def sample_gausian(df, cls, n, r):
  df = df.sample(n = n, random_state = r)
  return df

def sample_stratified(df, cls, n, r):
  n = min(n, df[cls].value_counts().min())
  df_ = df.groupby(cls).apply(lambda x: x.sample(n))
  df_.index = df_.index.droplevel(0)
  return df_

def load_data(df: None, args):
  # Load Dataset
  mode_intr_meth = args['mode_intr_meth']
  if mode_intr_meth == "trai":
    data_path = args["data_trai_path"]
    data_name = args["data_trai_name"]
    data_extn = args["data_trai_extn"]
    data_sepa = args["data_trai_sepa"]
    data_deci = args["data_trai_deci"]
  elif mode_intr_meth == "pred":
    data_path = args["data_pred_path"]
    data_name = args["data_pred_name"]
    data_extn = args["data_pred_extn"]
    data_sepa = args["data_pred_sepa"]
    data_deci = args["data_pred_deci"]

  data_mode = args["data_mode"]
  if data_mode == "local":
    df = pd.read_csv('{0}/{1}.{2}'.format(data_path, data_name, data_extn), sep=data_sepa, decimal=data_deci, low_memory=False)
  elif data_mode == "drive":
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    df = pd.read_csv('{0}/{1}.{2}'.format(data_path, data_name, data_extn), sep=data_sepa, decimal=data_deci, low_memory=False)
  elif data_mode == "dataframe":
    df = df

  data_smpl_mode = args["data_smpl_mode"]
  data_smpl_pop = args["data_smpl_pops"]
  if data_smpl_mode == True:
    data_smpl_ran = args["rands"]
    data_smpl_typ = args["data_smpl_type"]
    df_class = args['mode_pres_cols_clas']
    print("Sampling mode with {} samples".format(data_smpl_pop))
    if data_smpl_typ == "gausian":
      df = sample_gausian(df=df, cls=df_class, n = data_smpl_pop, r = data_smpl_ran)
    elif data_smpl_typ == "stratified":
      df = sample_stratified(df=df, cls=df_class, n = data_smpl_pop, r = data_smpl_ran)
  else:
    print("Complete mode with {} samples".format(data_smpl_pop))
  return df

def dict_list(list, key_name, key_value, val_name):
  for item in list:
    if item[key_name]==key_value:
      return item[val_name]

def detect_rows(df):
  #Check Row Lenght
  return len(df.index)

def detect_cols(df):
  #Check Col Lenght
  return len(df.columns)

def detect_shape(df):
  #Check Shape
  return df.shape

def detect_format(metric, value):
  if metric == "time":
    return "{}s".format(value)
  elif metric == "accuracy":
    return "{}%".format(value)
  elif metric == "r2":
    return "{}%".format(value)

def detect_sample(df, args):
  #Sampling Method
  args['data_smpl_mode'] = False
  args['data_smpl_pops'] = 0
  df = load_data(df, args)
  #------------------------------------#
  thrs_per = args["mode_pres_rows_thrs_per"]
  thrs_min = args["mode_pres_rows_thrs_min"]
  lens_alls = len(df)
  lens_thrd = int(lens_alls * thrs_per)
  #------------------------------------#
  if thrs_per == -1: #Manual Mode - Complete Dataset
    sample = {"smpl_mode": False, "smpl_pops": lens_alls }
  elif thrs_per == 0:
    sample = {"smpl_mode": True, "smpl_pops": thrs_min }
  elif thrs_per >= 0:
    if lens_alls <= thrs_min: #Automatic Mode - Dataset's length is smaller than sample ratio
      sample = {"smpl_mode": False, "smpl_pops": lens_alls }
    elif lens_alls > thrs_min: #Automatic Mode - Dataset's length is grater than sample ratio
      sample = {"smpl_mode": True, "smpl_pops": lens_thrd }
  return sample

def detect_types(df, args):
  df_types = pd.DataFrame(df.dtypes).reset_index().rename(columns={"index": "feature_name", 0: "feature_orig"})

  type_num = ["int16","int32","int64","float16","float32","float64"]
  type_str = ["string", "object"]
  type_cat = ["bool"]

  def transform(feature_name, feature_orig):
    if str.lower(str(feature_orig)) in type_num:
      df_types_thes = args['mode_pres_ftrs_thrs_typ']
      if (1.*df[feature_name].nunique()/df[feature_name].count() < df_types_thes): #or some other threshold
        return ["Numeric", "Categorical"]
      elif (1.*df[feature_name].nunique()/df[feature_name].count() >= df_types_thes): #or some other threshold
        return ["Numeric", "Numeric"]
    elif str.lower(str(feature_orig)) in type_str:
      df_types_thes = args['mode_pres_ftrs_thrs_typ']
      if (1.*df[feature_name].nunique()/df[feature_name].count() < df_types_thes): #or some other threshold
        return ["String", "Categorical"]
      elif (1.*df[feature_name].nunique()/df[feature_name].count() >= df_types_thes): #or some other threshold
        return ["String", "String"]
    elif str.lower(str(feature_orig)) in type_cat:
       return ["Bool", "Categorical"]
  df_types[["feature_type", "feature_ctgr"]] = df_types.apply(lambda x: transform(x.feature_name, x.feature_orig), axis=1, result_type='expand') #result_type=expand explodes the result
  return df_types

def detect_correlation(df, args):
  logs = True;
  df_types = args['intr_cols_type'] = detect_types(df, args)
  df_class = args['mode_pres_cols_clas']

  thrs_cor = args['mode_pres_cols_thrs_cor']
  thrs_min = args['mode_pres_cols_thrs_min']
  thrs_all = df.columns

  drop_cols = []
  if thrs_cor == -1: #Manual Mode - Complete Dataset
    drop_cols = []
  elif thrs_cor >= 0: #Automatic Mode
    df_tmp = pd.DataFrame()
    for col in df.columns:
      if (df_types[df_types["feature_name"] == col]["feature_type"].iloc[0]) == "String": #iloc=(0) vs iloc[0] has difference
        df_tmp[col] = df[col].astype('category').cat.codes
      else:
        df_tmp[col] = df[col]

    corr_matrix = df_tmp.drop(df_class, 1).corr()
    iters = range(len(corr_matrix.columns) - 1)

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
      for j in range(i+1):
        if (len(thrs_all) - len(drop_cols) -1) > thrs_min: #-1 for class which not counting as feature
          item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
          col = item.columns
          row = item.index
          val = abs(item.values)

          # If correlation exceeds the threshold
          if val >= thrs_cor:
            # Print the correlated features and the correlation value
            dropped_feature = ""
            if logs:
              print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
            col_value_corr = abs(df_tmp[col.values[0]].corr(df_tmp[df_class]))
            row_value_corr = abs(df_tmp[row.values[0]].corr(df_tmp[df_class]))
            if logs:
              print("{}: {}".format(col.values[0], np.round(col_value_corr, 3)))
              print("{}: {}".format(row.values[0], np.round(row_value_corr, 3)))
            if col_value_corr < row_value_corr:
              dropped_feature = "dropped: " + col.values[0]
              if col.values[0] not in drop_cols:
                drop_cols.append(col.values[0])
            else:
              dropped_feature = "dropped: " + row.values[0]
              if row.values[0] not in drop_cols:
                drop_cols.append(row.values[0])
            if logs:
                print(dropped_feature)
                print("-----------------------------------------------------------------------------")
  return drop_cols

def detect_features(df, args):
  df_types = args['intr_cols_type'] = detect_types(df, args)
  df_class = args["mode_pres_cols_clas"]

  df_types_nums = []
  df_types_strs = []
  df_types_cats = []

  df_types_nums_temp = df_types[(df_types["feature_name"] != df_class) & (df_types["feature_ctgr"] == "Numeric")]
  if df_types_nums_temp.empty == False:
    df_types_nums = df_types_nums_temp["feature_name"].tolist()
  df_types_strs_temp = df_types[(df_types["feature_name"] != df_class) & (df_types["feature_ctgr"] == "String")]
  if df_types_strs_temp.empty == False:
    df_types_strs = df_types_strs_temp["feature_name"].tolist()
  df_types_cats_temp = df_types[(df_types["feature_name"] != df_class) & (df_types["feature_ctgr"] == "Categorical")]
  if df_types_cats_temp.empty == False:
    df_types_cats = df_types_cats_temp["feature_name"].tolist()

  col_types = {'num_feature': df_types_nums, 'str_feature': df_types_strs, 'cat_feature': df_types_cats }
  return col_types

def detect_exist(df, classcol):
  if classcol in df.columns:
    exists = True
  else:
    exists = False
  return exists

def detect_minmax(df):
  #Boxplot Columns
  column_infos = []
  for feature in dataset_columns_nums:
    column_info = { "Feature": feature, "Min:": df[feature].min(), "Max": df[feature].max() }
    column_infos.append(column_info)
  return column_infos

def detect_cols_missing(df):
  def function(col):
    cols_all = col.fillna(value=0).count()
    cols_nna = col.isnull().sum()
    ratio = float(round((cols_nna / cols_all), 4))
    return ratio

  df_info_cols = pd.DataFrame(df.apply(function, axis=0), columns=["Ratio"])
  df_info_cols["Ratio Desc"] = df_info_cols["Ratio"].apply(lambda col: "{0}%".format(float(round(col * 100,2))))
  df_info_cols
  return df_info_cols

def detect_rows_missing(df):
  def function(row):
    cols_all = row.fillna(value=0).count()
    cols_nna = row.isnull().sum()
    ratio = float(round((cols_nna / cols_all), 4))
    return ratio

  df_info_rows = pd.DataFrame(df.apply(function, axis=1), columns=["Ratio"])
  df_info_rows["Ratio Desc"] = df_info_rows["Ratio"].apply(lambda row: "{0}%".format(float(round(row * 100,2))))
  df_info_rows
  return df_info_rows

def detect_area(df, args):
  df_types = detect_types(df, args)
  df_class = args["mode_pres_cols_clas"]
  df_types_clas = df_types[(df_types["feature_name"] == df_class)]["feature_ctgr"].iloc[0]

  mod_types = None
  if str(df_types_clas) == "String":
    mod_types = "classification"
  elif str(df_types_clas) == "Categorical":
    mod_types = "classification"
  elif str(df_types_clas) == "Numeric":
      mod_types = "regression"
  return mod_types

def detect_cardinality(df, args):
  #Check Class Cardinality
  dataset_columns_clas = args["mode_pres_cols_clas"]
  if dataset_columns_clas == "classification":
    cardinality = {}
    classes = df[dataset_columns_clas].unique()
    for iclass in classes:
      cardinality[iclass] = len(df[df[dataset_columns_clas] == iclass])
  elif dataset_columns_clas == "regression":
    ranges = df[dataset_columns_clas].unique()
    for irange in ranges:
      cardinality[irange] = len(df[df[dataset_columns_clas] == irange])
  return cardinality

def drop_xxxs(df, indexes, axis, args):
  df.drop(indexes, axis=axis, inplace=True)
  return df

def data_remove(df, selected_columns_xxxs, mode_type):
  df[selected_columns_xxxs] = df[selected_columns_xxxs].dropna(axis=0)
  return df[selected_columns_xxxs]

def data_replace(df, selected_columns_xxxs, mode_type):
  if mode_type in ('nums'):
    df[selected_columns_xxxs] = df[selected_columns_xxxs].fillna(value=0)
  elif mode_type in ('strs'):
    df[selected_columns_xxxs] = df[selected_columns_xxxs].fillna(value='-None-')
  elif mode_type in ('cats'):
    df[selected_columns_xxxs] = df[selected_columns_xxxs].fillna(value='-None-')
  return df[selected_columns_xxxs]

def data_impute(df, dataset_columns_nums, dataset_columns_strs, dataset_columns_cats, mode_type, args):
  njobs = args["njobs"]
  rands = args["rands"]

  rd.seed(rands)
  np.random.seed(rands)

  mode_labl = args["mode_pres_rows_labl"]
  
  dataset_columns_alls = []
  dataset_columns_alls.extend(dataset_columns_nums)
  dataset_columns_alls.extend(dataset_columns_strs)
  dataset_columns_alls.extend(dataset_columns_cats)

  filled_cols = []
  missed_cols = []
  for feature in dataset_columns_alls:
    df[feature + '_imp'] = df[feature]
    number_missing_alls = df[feature].isnull().sum()
    if number_missing_alls == 0:
      filled_cols.append(feature)
    else:
      missed_cols.append(feature)
      observed_values_alls = df.loc[df[feature].notnull(), feature]
      df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values_alls, number_missing_alls, replace = True)

  if mode_type == 'nums':
    dataset_columns_xxxs = dataset_columns_nums
    df_det_xxxs = pd.DataFrame(columns = dataset_columns_xxxs)
  elif mode_type == 'strs':
    dataset_columns_xxxs = dataset_columns_strs
    df_det_xxxs = pd.DataFrame(columns = dataset_columns_xxxs)
  elif mode_type == 'cats':
    dataset_columns_xxxs = dataset_columns_cats
    df_det_xxxs = pd.DataFrame(columns = dataset_columns_xxxs)

  ####################################################
  impute_cols = OrderedSet(dataset_columns_xxxs)-OrderedSet(missed_cols)
  for feature in impute_cols:
    print("Copying feature: {0}".format(feature))
    df_det_xxxs[feature] = df[feature]
  ####################################################
  impute_cols = OrderedSet(dataset_columns_xxxs)-OrderedSet(filled_cols)
  for feature in impute_cols:
    print("Imputing feature: {0}".format(feature))
    df_det_xxxs[feature] = df[feature + "_imp"]

    parameters_nums = list(OrderedSet([feature + '_imp' for feature in dataset_columns_nums]) - OrderedSet([feature + '_imp']))
    parameters_strs = list(OrderedSet([feature + '_imp' for feature in dataset_columns_strs]) - OrderedSet([feature + '_imp']))
    parameters_cats = list(OrderedSet([feature + '_imp' for feature in dataset_columns_cats]) - OrderedSet([feature + '_imp']))
    parameters_alls = parameters_nums + parameters_strs + parameters_cats

    count = 0
    for feature_inp in parameters_alls:
      encoder = None
      feature_dat = None
      ####################################################
      if feature_inp in parameters_nums:
        if count == 0:
          features = df[feature_inp]
        else:
          features = pd.concat([features, df[feature_inp]], axis=1)
        count = count + 1
      ####################################################
      elif feature_inp in parameters_strs:
        if mode_labl == "label":
          encoder = LabelEncoder()
          feature_dat = [[str(s)] for s in df[feature_inp]]
          feature_enc = pd.DataFrame(encoder.fit_transform(feature_dat), columns=[feature_inp], index=df.index)
        elif mode_labl == "multi":
          encoder = MultiLabelBinarizer()
          feature_dat = [[str(s)] for s in df[feature_inp]]
          feature_enc = pd.DataFrame(encoder.fit_transform(feature_dat), index=df.index)
        if count == 0:
          features = feature_enc
        else:
          features = pd.concat([features, feature_enc], axis=1)
        count = count + 1
      ####################################################
      elif feature_inp in parameters_cats:
        if mode_labl == "label":
          encoder = LabelEncoder()
          feature_dat = [[str(s)] for s in df[feature_inp]]
          feature_enc = pd.DataFrame(encoder.fit_transform(feature_dat), columns=[feature_inp], index=df.index)
        elif mode_labl == "multi":
          encoder = MultiLabelBinarizer()
          feature_dat = [[str(s)] for s in df[feature_inp]]
          feature_enc = pd.DataFrame(encoder.fit_transform(feature_dat), index=df.index)
        if count == 0:
          features = feature_enc
        else:
          features = pd.concat([features, feature_enc], axis=1)
        count = count + 1
      ####################################################
    featurest = df[feature + '_imp'].astype(str)

    if feature in dataset_columns_nums:
      #Create a Linear Regression model to estimate the missing data
      model = LinearRegression(n_jobs=njobs)
      model.fit(X = features, y = featurest)
    elif feature in dataset_columns_strs:
      if (len(featurest.unique()) > 1):
        #Create a Logistic Regression model to estimate the missing data
        model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=rands, n_jobs=njobs)
        model.fit(X = features, y = featurest)
      else:
        #Create a Random Forest model to estimate the missing data
        model = RandomForestClassifier(random_state=rands, n_jobs=njobs)
        model.fit(X = features, y = featurest)
    elif feature in dataset_columns_cats:
      if (len(featurest.unique()) > 1):
        #Create a Logistic Regression model to estimate the missing data
        model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=rands, n_jobs=njobs)
        model.fit(X = features, y = featurest)
      else:
        #Create a Random Forest model to estimate the missing data
        model = RandomForestClassifier(random_state=rands, n_jobs=njobs)
        model.fit(X = features, y = featurest)
    
    #observe that I preserve the index of the missing data from the original dataframe
    df_det_xxxs.loc[df[feature].isnull(), feature] = model.predict(features)[df[feature].isnull()]
  return df_det_xxxs

def data_manage(df, args, feats):
  mode_intr_meth = args['mode_intr_meth']
  if mode_intr_meth == "trai":
    #Mode of handling Cols
    print("====================== Cols Selected ======================")
    mode_pres_cols = args['mode_pres_cols']
    print("Method: {0}".format(mode_pres_cols))
    print("-----------------------------------")

    if args['intr_keep_cols'] == 0:
      if mode_pres_cols == "choose":
        selected_columns_nums = list(OrderedSet(args["mode_pres_cols_nums_chs"]))
        selected_columns_strs = list(OrderedSet(args["mode_pres_cols_strs_chs"]))
        selected_columns_cats = list(OrderedSet(args["mode_pres_cols_cats_chs"]))
        selected_columns_clas = args["mode_pres_cols_clas"]
      elif mode_pres_cols == "substruct":
        selected_columns_nums = list(OrderedSet(args["mode_pres_cols_nums_all"]) - OrderedSet(args["mode_pres_cols_nums_sub"]))
        selected_columns_strs = list(OrderedSet(args["mode_pres_cols_strs_all"]) - OrderedSet(args["mode_pres_cols_strs_sub"]))
        selected_columns_cats = list(OrderedSet(args["mode_pres_cols_cats_all"]) - OrderedSet(args["mode_pres_cols_cats_sub"]))
        selected_columns_clas = args["mode_pres_cols_clas"]
      elif mode_pres_cols == "complete":
        selected_columns_nums = list(OrderedSet(args["mode_pres_cols_nums_all"]))
        selected_columns_strs = list(OrderedSet(args["mode_pres_cols_strs_all"]))
        selected_columns_cats = list(OrderedSet(args["mode_pres_cols_cats_all"]))
        selected_columns_clas = args["mode_pres_cols_clas"]
      elif mode_pres_cols == "auto":
        args['intr_cols_miss'] = detect_cols_missing(df)
        print("Cols Information:")
        print(args['intr_cols_miss'] )
        print("-----------------------------------")
        thes = args["mode_pres_ftrs_thrs_mis"]
        drop_cols = args['intr_cols_miss'][args['intr_cols_miss']["Ratio"] > thes].index.tolist()
        print("Cols Deleted: {0}".format(drop_cols))
        df = drop_xxxs(df, drop_cols, 1, args)
        print("-----------------------------------")

        args['intr_cols_corr'] = detect_correlation(df, args)
        print("Column correlated: {0}".format(args['intr_cols_corr']))
        df = drop_xxxs(df, args['intr_cols_corr'], 1, args)
        print("-----------------------------------")

        result = detect_features(df, args)
        selected_columns_nums = result['num_feature']
        selected_columns_strs = result['str_feature']
        selected_columns_cats = result['cat_feature']
        selected_columns_clas = args["mode_pres_cols_clas"]

    elif args['intr_keep_cols'] == 1:
      selected_columns_nums = feats['num_feature']
      selected_columns_strs = feats['str_feature']
      selected_columns_cats = feats['cat_feature']
      selected_columns_clas = args["mode_pres_cols_clas"]

    print("Numeric: {0}".format(selected_columns_nums))
    print("Descriptive: {0}".format(selected_columns_strs))
    print("Categorical: {0}".format(selected_columns_cats))
    print("============================================================")

    #Mode of handling Rows
    print("====================== Rows Selected ======================")
    mode_pres_rows = args['mode_pres_rows']
    print("Method: {0}".format(mode_pres_rows))
    print("-----------------------------------")

    if mode_pres_rows == "remove":
      df_nums = data_remove(df, selected_columns_nums, 'nums')
      df_strs = data_remove(df, selected_columns_strs, 'strs')
      df_cats = data_remove(df, selected_columns_cats, 'cats')
      df_clas = df[selected_columns_clas]
    elif mode_pres_rows == "replace":
      df_nums = data_replace(df, selected_columns_nums, 'nums')
      df_strs = data_replace(df, selected_columns_strs, 'strs')
      df_cats = data_replace(df, selected_columns_cats, 'cats')
      df_clas = df[selected_columns_clas]
    elif mode_pres_rows == "impute":
      df_nums = data_impute(df, selected_columns_nums, selected_columns_strs, selected_columns_cats, 'nums', args)
      df_strs = data_impute(df, selected_columns_nums, selected_columns_strs, selected_columns_cats, 'strs', args)
      df_cats = data_impute(df, selected_columns_nums, selected_columns_strs, selected_columns_cats, 'cats', args)
      df_clas = df[selected_columns_clas]
    elif mode_pres_rows == "auto":
      df_rows_info = detect_rows_missing(df)
      print("Rows Information:")
      print(df_rows_info)
      print("-----------------------------------")
      thes = args["mode_pres_ftrs_thrs_mis"]
      drop_rows = df_rows_info[df_rows_info["Ratio"] > thes].index.tolist()
      print("Rows Deleted: {0}".format(drop_rows))
      df = drop_xxxs(df, drop_rows, 0, args)
      print("-----------------------------------")
      df_nums = data_impute(df, selected_columns_nums, selected_columns_strs, selected_columns_cats, 'nums', args)
      df_strs = data_impute(df, selected_columns_nums, selected_columns_strs, selected_columns_cats, 'strs', args)
      df_cats = data_impute(df, selected_columns_nums, selected_columns_strs, selected_columns_cats, 'cats', args)
      df_clas = df[selected_columns_clas]

    print("-----------------------------------")
    print("Numeric: {0}".format(selected_columns_nums))
    print("Descriptive: {0}".format(selected_columns_strs))
    print("Categorical: {0}".format(selected_columns_cats))
    print("============================================================")

  elif mode_intr_meth == "pred":
    print("====================== Cols Selected ======================")
    selected_columns_nums = feats['num_feature']
    selected_columns_strs = feats['str_feature']
    selected_columns_cats = feats['cat_feature']
    if detect_exist(df, args["mode_pres_cols_clas"]):
      selected_columns_clas = args["mode_pres_cols_clas"]
    else:
      selected_columns_clas = None
    print("Numeric: {0}".format(selected_columns_nums))
    print("Descriptive: {0}".format(selected_columns_strs))
    print("Categorical: {0}".format(selected_columns_cats))

    df_nums = data_impute(df, selected_columns_nums, selected_columns_strs, selected_columns_cats, 'nums', args)
    df_strs = data_impute(df, selected_columns_nums, selected_columns_strs, selected_columns_cats, 'strs', args)
    df_cats = data_impute(df, selected_columns_nums, selected_columns_strs, selected_columns_cats, 'cats', args)
    if detect_exist(df, args["mode_pres_cols_clas"]):
      df_clas = df[selected_columns_clas]
    else:
      df_clas = None

  data_feats = {
      "num_feature": selected_columns_nums,
      "str_feature": selected_columns_strs,
      "cat_feature": selected_columns_cats,
      "all_feature": selected_columns_nums + selected_columns_strs + selected_columns_cats
  }
  return df_nums, df_strs, df_cats, df_clas, data_feats

def model_manage(df, models, args):
  #Mode of handling areas
  print("====================== Area Selected ======================")
  mode_pres_type = args['mode_pres_type']
  print("Method: {0}".format(mode_pres_type))
  mode_pres_type = args["mode_pres_type"]
  if mode_pres_type == "auto":
    models_area = detect_area(df, args)
  else:
    models_area = mode_pres_type
  args['intr_mode_area'] = models_area
  print("Area: {0}".format(models_area))
  print("============================================================")

  #Mode of handling models
  print("====================== Models Selected ======================")
  models_avail = models
  models_prefe = args['mode_pres_mdls']
  models_selec_objt = []
  models_selec_name = []
  for model in models_avail:
    if (model["id"] in models_prefe) & (model["type"] == models_area):
      models_selec_objt.append(model)
      models_selec_name.append(type(model["model"]).__name__)
  print(models_selec_name)
  print("============================================================")
  return models_selec_objt

def preprocessing(df_nums, df_strs, df_cats, df_clas, args, encos):
  mode_intr_meth = args['mode_intr_meth']

  ###NUMERIC FEATURES###
  count_cols_num = 0
  for feature in df_nums.columns:
    if count_cols_num == 0:
      features_num_red = df_nums[feature]
    else:
      features_num_red = pd.concat([features_num_red, df_nums[feature]], axis=1)
    count_cols_num = count_cols_num + 1

  ###NOMINAL FEATURES###
  count_cols_str = 0
  str_encoders = []
  for feature in df_strs.columns:
    feature_dat = [[str(s)] for s in df_strs[feature]]
    if args['mode_intr_meth'] == "trai":
      if args["mode_prep_labl"] == "label":
        str_encoder = LabelEncoder()
        feature_enc = pd.DataFrame(str_encoder.fit_transform(feature_dat), columns=[feature], index=df_strs.index)
      elif args["mode_prep_labl"] == "multi":
        str_encoder = MultiLabelBinarizer()
        feature_enc = pd.DataFrame(str_encoder.fit_transform(feature_dat), index=df_strs.index)
      str_encoder = {'feature': feature, 'encoder': str_encoder }
      str_encoders.append(str_encoder)
    elif args['mode_intr_meth'] == "pred":
      str_encoders = encos["str_encoders"]
      str_encoder = dict_list(str_encoders, "feature", feature, "encoder")
      if args["mode_prep_labl"] == "label":
        feature_enc = pd.DataFrame(str_encoder.transform(feature_dat), columns=[feature], index=df_strs.index)
      elif args["mode_prep_labl"] == "multi":
        feature_enc = pd.DataFrame(str_encoder.transform(feature_dat), index=df_strs.index)
    if count_cols_str == 0:
      features_str_red = feature_enc
    else:
      features_str_red = pd.concat([features_str_red, feature_enc], axis=1)
    count_cols_str = count_cols_str + 1

  ###CATEGORICAL FEATURES###
  count_cols_cat = 0
  cat_encoders = []
  for feature in df_cats.columns:

    feature_dat = [[str(s)] for s in df_cats[feature]]
    if args['mode_intr_meth'] == "trai":
      if args["mode_prep_labl"] == "label":
        cat_encoder = LabelEncoder()
        feature_enc = pd.DataFrame(cat_encoder.fit_transform(feature_dat), columns=[feature], index=df_cats.index)
      elif args["mode_prep_labl"] == "multi":
        cat_encoder = MultiLabelBinarizer()
        feature_enc = pd.DataFrame(cat_encoder.fit_transform(feature_dat), index=df_cats.index)
      cat_encoder = { 'feature': feature, 'encoder': cat_encoder }
      cat_encoders.append(cat_encoder)
    elif args['mode_intr_meth'] == "pred":
      cat_encoders = encos["cat_encoders"]
      cat_encoder = dict_list(cat_encoders, "feature", feature, "encoder")
      if args["mode_prep_labl"] == "label":
        feature_enc = pd.DataFrame(cat_encoder.transform(feature_dat), columns=[feature], index=df_cats.index)
      elif args["mode_prep_labl"] == "multi":
        feature_enc = pd.DataFrame(cat_encoder.transform(feature_dat), index=df_cats.index)
    if count_cols_cat == 0:
      features_cat_red = feature_enc
    else:
      features_cat_red = pd.concat([features_cat_red, feature_enc], axis=1)
    count_cols_cat = count_cols_cat + 1

  ###ALL FEATURES###
  X = None
  if count_cols_num > 0 and count_cols_str > 0 and count_cols_cat > 0:
    X = features_num_red
    X = pd.concat([X, features_str_red], axis=1)
    X = pd.concat([X, features_cat_red], axis=1)
  elif count_cols_num > 0 and count_cols_str > 0:
    X = pd.concat([features_num_red, features_str_red], axis=1)
  elif count_cols_num > 0 and count_cols_cat > 0:
    X = pd.concat([features_num_red, features_cat_red], axis=1)
  elif count_cols_str > 0 and count_cols_cat > 0:
    X = pd.concat([features_str_red, features_cat_red], axis=1)
  elif count_cols_num > 0:
    X = features_num_red
  elif count_cols_str > 0:
    X = features_str_red
  elif count_cols_cat > 0:
    X = features_cat_red

  ###ALL CLASSES###
  y = df_clas

  #FEATURE COUNT#
  if X is None:
    features_columns = 0
    features_name = ""
    args["intr_data_type"] = "empty"
  elif isinstance(X, pd.Series):
    features_columns = 1
    features_name = X.name
    args["intr_data_type"] = "series"
    X = X.values.reshape(-1,1)
    y = y.values.reshape(-1,1)
  elif isinstance(X, pd.DataFrame):
    args["intr_data_type"] = "frame"
    features_name = X.columns
    features_columns = len(X.columns)

  #COLUMN NAMES
  class_name = [args["mode_pres_cols_clas"]]

  #BALANCE DATASET#
  mode_prep_bala_method = args['mode_prep_bala_method']
  if mode_prep_bala_method in ["all", "over"]:
    if mode_intr_meth == "trai":
      oversample = SMOTE()
      X, y = oversample.fit_resample(X.values, y)
      X = pd.DataFrame(X, columns=features_name)
      y = pd.DataFrame(y, columns=class_name)

  #BALANCE DATASET#
  mode_prep_bala_method = args['mode_prep_bala_method']
  if mode_prep_bala_method in ["all", "under"]:
    if mode_intr_meth == "trai":
      undersample = NeighbourhoodCleaningRule(n_neighbors=3, threshold_cleaning=0.5)
      X, y = undersample.fit_resample(X.values, y)
      X = pd.DataFrame(X, columns=features_name)
      y = pd.DataFrame(y, columns=class_name)

  #SCALE DATASET#
  scaler = None
  mode_prep_sclr_method = args['mode_prep_sclr_method']
  if mode_prep_sclr_method == "standard":
    scaler = StandardScaler()
    if mode_intr_meth == "trai":
      scaler.fit(X)
    elif mode_intr_meth == "pred":
      scaler = encos["scr_encoders"]
      X = scaler.transform(X)

  #REDUCE DIMENSIONALITY#
  reducer = None
  mode_prep_redu_method = args['mode_prep_redu_method']
  mode_prep_redu_compon = args['mode_prep_redu_compon']
  #PCA UNSUPERVISED LINEAR
  if mode_prep_redu_method == "pca":
    if mode_intr_meth == "trai":
      reducer = PCA(n_components=mode_prep_redu_compon)
      X = reducer.fit_transform(X)
    elif mode_intr_meth == "pred":
      reducer = encos["rdr_encoders"]
      X = reducer.transform(X)
  #SVD UNSUPERVISED LINEAR
  if mode_prep_redu_method == "svd":
    if mode_intr_meth == "trai":
      reducer = TruncatedSVD(n_components=mode_prep_redu_compon)
      X = reducer.fit_transform(X)
    elif mode_intr_meth == "pred":
      reducer = encos["rdr_encoders"]
      X = reducer.transform(X)
  #KPCA UNSUPERVISED NON-LINEAR
  if mode_prep_redu_method == "kpca":
    if mode_intr_meth == "trai":
      reducer = KernelPCA(n_components=mode_prep_redu_compon)
      X = reducer.fit_transform(X)
    elif mode_intr_meth == "pred":
      reducer = encos["rdr_encoders"]
      X = reducer.transform(X)
  #KNN UNSUPERVISED NON-LINEAR
  if mode_prep_redu_method == "kneig":
    if mode_intr_meth == "trai":
      reducer = NeighborhoodComponentsAnalysis(n_components=mode_prep_redu_compon)
      X = reducer.fit_transform(X)
    elif mode_intr_meth == "pred":
      reducer = encos["rdr_encoders"]
      X = reducer.transform(X)
  #LDA SUPERVISED LINEAR
  if mode_prep_redu_method == "discr":
    if mode_intr_meth == "trai":
      reducer = LinearDiscriminantAnalysis(n_components=mode_prep_redu_compon)
      X = reducer.fit_transform(X, y)
    elif mode_intr_meth == "pred":
      reducer = encos["rdr_encoders"]
      X = reducer.transform(X, y)

  count_rows_all = len(X)
  count_cols_all = count_cols_num + count_cols_str + count_cols_cat

  data_specs = {
      "model_rows_all": count_rows_all,
      "model_cols_num": count_cols_num,
      "model_cols_str": count_cols_str,
      "model_cols_cat": count_cols_cat,
      "model_cols_all": count_cols_all
    }
  data_encos = {
      "str_encoders": str_encoders,
      "cat_encoders": cat_encoders,
      "scr_encoders": scaler,
      "rdr_encoders": reducer
    }
  return X, y, data_specs, data_encos

def modeling(models, data_specs):
  dict_modeling_infos = []
  for clas_model in models:
    model_name = type(clas_model["model"]).__name__
    dict_info = {
      "model_name": model_name,
      "model_rows_all": data_specs["model_rows_all"],
      "model_cols_num": data_specs["model_cols_num"],
      "model_cols_str": data_specs["model_cols_str"],
      "model_cols_cat": data_specs["model_cols_cat"],
      "model_cols_all": data_specs["model_cols_all"]
    }
    dict_modeling_infos.append(dict_info)

  df_models = pd.DataFrame(dict_modeling_infos)
  df_models = df_models.set_index('model_name')

  return df_models

def space_override(model, args, mode):
  model_name = type(model["model"]).__name__
  if mode == "none":
    if model_name == "KNeighborsRegressor":
      folds = args['folds']
      model["model"].n_neighbors = folds - 1
  elif mode == "bayssian":
    folds = args['folds']
    if model_name == "KNeighborsRegressor":
      model["search_space"]["n_neighbors"] = [folds - 1]
  return model

def sampling(models, X, y, args):
  dict_training_infos = []
  rands = args['rands']
  folds = args['folds']
  metrc = args['mode_eval_mtrc']
  njobs = args['mode_eval_njob']
  niter = args['mode_eval_nite']
  dtype = args["intr_data_type"]
  drows = len(X)

  intr_mode_area = args['intr_mode_area']
  if metrc == "auto":
    if intr_mode_area == "classification":
      args['intr_eval_mtrc'] = "accuracy"
    elif intr_mode_area == "regression":
      args['intr_eval_mtrc'] = "r2"
  else:
    args['intr_eval_mtrc'] = metrc

  print("====================== Model Performance ======================")
  for model in models:
    rd.seed(rands)
    np.random.seed(rands)
    model_idnm = type(model["id"]).__name__
    model_name = type(model["model"]).__name__
    print("----------------------------------------------------")
    print("Training_Model: {}".format(model_name))

    time_str = tm.time()
    #############################################################################################################################################################
    model_init = None
    model_objt = None
    metrc = args['intr_eval_mtrc']
    mode_eval_meth = args["mode_eval_meth"]

    ###MODEL OVD###
    model = space_override(model, args, mode_eval_meth)
    if mode_eval_meth == "none":
      ###MODEL INI###
      model_objt = model["model"]
      ###MODEL FIT###
      if dtype == "frame":
        model_init.fit(X, y.values.ravel())
      else:
        model_init.fit(X, y)
      ###MODEL SCR###
      model_scra = model_init.score(X, y) #Returns score on known data / seen data
      model_scrb = model_init.score(X, y) #Returns score on unknown data / unseen data
      model_spac = None
    elif mode_eval_meth == "grid":
      ###MODEL INI###
      from sklearn.model_selection import GridSearchCV
      model_init = GridSearchCV(model["model"], model["search_space"], scoring=metrc, n_jobs=njobs, cv=folds)
      model_objt = GridSearchCV(model["model"], model["search_space"], scoring=metrc, n_jobs=njobs, cv=folds)
    elif mode_eval_meth == "random":
      ###MODEL INI###
      from sklearn.model_selection import RandomizedSearchCV
      model_init = RandomizedSearchCV(model["model"], model["search_space"], scoring=metrc, n_iter=niter, n_jobs=njobs, cv=folds, random_state=rands)
      model_objt = RandomizedSearchCV(model["model"], model["search_space"], scoring=metrc, n_iter=niter, n_jobs=njobs, cv=folds, random_state=rands)
    elif mode_eval_meth == "bayssian":
      ###MODEL INI###
      model_init = BayesSearchCV(model["model"], model["search_space"], scoring=metrc, n_iter=niter, n_jobs=njobs, cv=folds, random_state=rands, return_train_score=True)
      model_objt = BayesSearchCV(model["model"], model["search_space"], scoring=metrc, n_iter=niter, n_jobs=njobs, cv=folds, random_state=rands, return_train_score=True)
      ###MODEL FIT###
      if dtype == "frame":
        model_init.fit(X, y.values.ravel())
      else:
        model_init.fit(X, y)
      model_indx = model_init.best_index_
      model_rlts = pd.DataFrame(model_init.cv_results_)
      ###MODEL SCR###
      model_scra = round(model_rlts['mean_train_score'][model_indx], 2) #Returns score on known data / seen data
      model_scrb = round(model_rlts['mean_test_score'][model_indx], 2) #Returns score on unknown data / unseen data
      model_spac = model_init.best_params_
    #############################################################################################################################################################
    time_end = tm.time()

    if intr_mode_area in ["classification", "regression"]:
      model_dura = round(time_end - time_str, 2)

      dict_info = {
        "model_idnm": model_idnm,
        "model_name": model_name,
        "model_init": model_init,
        "model_objt": model_objt,
        "model_mtrc": metrc,
        "model_scra": model_scra,
        "model_scrb": model_scrb,
        "model_dura": model_dura,
        "model_spac": model_spac
      }
      dict_training_infos.append(dict_info)

    print("Training_Score: {}".format(detect_format(metrc, model_scra)))
    print("Testing_Score: {}".format(detect_format(metrc, model_scrb)))
    print("Process_Duration: {}".format(detect_format("time", model_dura)))
    print("Process_Parameters: {}".format(model_spac))
    df_models = pd.DataFrame(dict_training_infos)
    df_models = df_models.set_index('model_name')
  return df_models

def find(models, data_specs, X, y, args):
  dict_modeling_infos = modeling(models, data_specs)
  dict_performa_infos = sampling(models, X, y, args)
  dict_models_infos = dict_modeling_infos.join(dict_performa_infos)
  #-------------------------------#
  dict_columns = []
  dict_columns.extend(dict_modeling_infos.keys())
  dict_columns.append("model_ordr")
  dict_columns.extend(dict_performa_infos.keys())
  dict_columns.remove("model_idnm")
  dict_columns.remove("model_init")
  dict_columns.remove("model_objt")
  #-------------------------------#
  df_models_infos = pd.DataFrame(dict_models_infos)
  df_models_infos["model_ordr"] = df_models_infos.sort_values(["model_scrb", "model_dura"], ascending=[False,True]).groupby(["model_idnm"]).cumcount() + 1
  #-------------------------------#
  print("====================== Model Evaluation ======================")
  print(df_models_infos[dict_columns].to_string())
  return df_models_infos

def best(df_models, sample):
  df_models = df_models[df_models["model_scrb"] == df_models["model_scrb"].max()]
  df_models = df_models[df_models["model_dura"] == df_models["model_dura"].min()]
  best_model = df_models.reset_index(level=0).iloc[0]
  best_model_name = best_model["model_name"]
  best_model_init = best_model["model_init"]
  best_model_objt = best_model["model_objt"]
  best_model_mtrc = best_model["model_mtrc"]
  best_model_scra = best_model["model_scra"]
  best_model_scrb = best_model["model_scrb"]
  best_model_spac = best_model["model_spac"]
  best_model_dura = best_model["model_dura"]
  print("====================== Model Selection ======================")
  print("Model Name: {0}".format(best_model_name))
  print("Model Training Score: {0}".format(detect_format(best_model_mtrc, best_model_scra)))
  print("Model Testing Score: {0}".format(detect_format(best_model_mtrc, best_model_scrb)))
  print("Model Process Duration: {0}".format(detect_format("time", best_model_dura)))
  print("Model Process Parameters: {0}".format(best_model_spac))

  best_model = None
  if sample == True:
    best_model = best_model_objt
  else:
    best_model = best_model_init
  return best_model

def trai(model, X, y, args):
  time_str = tm.time()
  dtype = args["intr_data_type"]
  if dtype == "frame":
    model.fit(X, y.values.ravel())
  else:
    model.fit(X, y)
  time_end = tm.time()

  model_name = type(model.estimator).__name__
  model_indx = model.best_index_
  model_rlts = pd.DataFrame(model.cv_results_)
  model_scra = round(model_rlts['mean_train_score'][model_indx], 2) #Returns score on known data / seen data
  model_scrb = round(model_rlts['mean_test_score'][model_indx], 2) #Returns score on unknown data / unseen data 
  model_dura = round(time_end - time_str, 2)
  model_mtrc = args['intr_eval_mtrc']
  model_spac = model.best_params_
  print("====================== Complete Training ======================")
  print("Model Name: {0}".format(model_name))
  print("Model Training Score: {0}".format(detect_format(model_mtrc, model_scra)))
  print("Model Testing Score: {0}".format(detect_format(model_mtrc, model_scrb)))
  print("Model Process Duration: {0}".format(detect_format("time", model_dura)))
  print("Model Process Parameters: {0}".format(model_spac))
  return model

def pred(model, X, y, args):
  y_hat = model.predict(X)
  y_act = y
  return y_hat, y_act

def auto_ml_train(df: None, args):
  args['mode_intr_meth'] = "trai"

  #Sampling Method
  sample = detect_sample(df, args)
  args['data_smpl_mode'] = sample["smpl_mode"]
  args['data_smpl_pops'] = sample["smpl_pops"]

  args['intr_keep_cols'] = 0
  args['intr_keep_rows'] = 0

  drop_cols = args['mode_pres_drop_cols']
  drop_rows = args['mode_pres_drop_rows']

  df_tmp = df
  df_tmp = load_data(df_tmp, args)
  df_tmp = drop_xxxs(df_tmp, drop_cols, 1, args)
  df_tmp = drop_xxxs(df_tmp, drop_rows, 0, args)
  df_nums, df_strs, df_cats, df_clas, data_feats = data_manage(df_tmp, args, None)
  X, y, data_specs, data_encos = preprocessing(df_nums, df_strs, df_cats, df_clas, args, None)

  #Modeling Method
  models = load_models(args)
  models = model_manage(df_tmp, models, args)
  models = find(models, data_specs, X, y, args)
  modelx = best(models, sample["smpl_mode"])

  #Complete Method
  if sample["smpl_mode"] == True:
    args['data_smpl_mode'] = False
    args['data_smpl_pops'] = 0

    args['intr_keep_cols'] = 1
    args['intr_keep_rows'] = 1

    drop_cols = args['mode_pres_drop_cols']
    drop_rows = args['mode_pres_drop_rows']

    df_tmp = df
    df_tmp = load_data(df_tmp, args)
    df_tmp = drop_xxxs(df_tmp, drop_cols, 1, args)
    df_tmp = drop_xxxs(df_tmp, drop_rows, 0, args)
    df_nums, df_strs, df_cats, df_clas, data_feats = data_manage(df_tmp, args, data_feats)
    X, y, data_specs, data_encos = preprocessing(df_nums, df_strs, df_cats, df_clas, args, data_encos)

    #With Refit Method
    model = trai(modelx, X, y, args)
  else:
    #Without Refit Method
    model = modelx

  model_args = {'model': model, 'feats': data_feats, 'encos': data_encos }
  return model_args

def auto_ml_pred(df: None, args, model_args):
    args['mode_intr_meth'] = "pred"

    args['data_smpl_mode'] = False
    args['data_smpl_pops'] = 0

    drop_cols = args['mode_pres_drop_cols']
    drop_rows = args['mode_pres_drop_rows']

    df_tmp = df
    df_tmp = load_data(df_tmp, args)
    df_tmp = drop_xxxs(df_tmp, drop_cols, 1, args)
    df_tmp = drop_xxxs(df_tmp, drop_rows, 0, args)

    feats = model_args["feats"]
    df_nums, df_strs, df_cats, df_clas, data_feats = data_manage(df_tmp, args, feats)

    encos = model_args["encos"]
    X, y, data_specs, data_encos = preprocessing(df_nums, df_strs, df_cats, df_clas, args, encos)

    #Pred Method
    model = model_args["model"]
    y_hat, y_act = pred(model, X, y, args)
    return y_hat, y_act
