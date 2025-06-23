# model training and hyperparamter tunings

# from project_config import RANDOM_STATE, N_ITER_SEARCH
import logging
import joblib
from datetime import datetime
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from preprocessor import get_preprocessor
from data_preparation import prepare_data
import custom_cv
import pandas as pd
from sklearn.metrics import make_scorer, f1_score
import warnings
import streamlit as st

N_ITER_SEARCH = int(st.secrets["N_ITER_SEARCH"])
RANDOM_STATE = int(st.secrets["RANDOM_STATE"])

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)


def initialize_models(preprocessor):
    models = {

        'LogisticRegression': {
            'pipeline': ImbPipeline(steps = [
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=RANDOM_STATE)),  # or ADASYN, SMOTETomek
                ('classifier', LogisticRegression(
                    random_state=RANDOM_STATE,
                    class_weight='balanced',
                    warm_start=True,
                    solver='saga'
                ))
            ]),
            'search_spaces': {
                'classifier__penalty': Categorical(['l1']),
                'classifier__C': Real(0.01, 5.0, prior='log-uniform'),
                'classifier__tol': Real(1e-4, 1e-3, prior='log-uniform'),
                'classifier__max_iter': Integer(5000, 20000),
                'classifier__n_jobs': Categorical([-1]),
                'smote__k_neighbors': Integer(1, 5),
                'smote__sampling_strategy': Categorical(['auto', 0.5, 0.75])
            }
        },

        'RandomForestClassifier': {
            'pipeline': ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('feature_selection', SelectFromModel(
                    estimator=RandomForestClassifier(
                        random_state=RANDOM_STATE,
                        class_weight='balanced',
                        n_jobs=-1,
                        n_estimators=100, 
                        max_depth=10 
                    ),
                    prefit=False,
                )),
                ('classifier', RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1))
            ]),
            'search_spaces': {
                # Classifier Hyperparameters
                'classifier__n_estimators': Integer(250, 700),
                'classifier__max_depth': Integer(15, 25),
                'classifier__min_samples_leaf': Integer(25, 60), 
                'classifier__min_samples_split': Integer(20, 40), 
                'classifier__max_features': Categorical(['sqrt', 'log2', 0.4, 0.5, 0.6]),
                'classifier__ccp_alpha': Real(0.005, 0.05, prior='log-uniform'),

                # SMOTE Hyperparameters 
                'smote__k_neighbors': Integer(3, 7), 
                'smote__sampling_strategy': Categorical([0.25, 0.5, 0.75]), 

                # Feature Selection (SelectFromModel) Hyperparameters
                'feature_selection__max_features': Integer(10, 35), 
            }
        },

        'Bagging_Classifier': {
            'pipeline': ImbPipeline(steps = [
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state = RANDOM_STATE)),
                ('feature_selection', SelectFromModel(
                    estimator=RandomForestClassifier(
                        random_state=RANDOM_STATE,
                        class_weight='balanced',
                        n_jobs=-1,
                        n_estimators=100,
                        max_depth=10
                    ),
                    prefit=False,
                )),
                ('classifier', BaggingClassifier(
                    estimator = DecisionTreeClassifier(class_weight = 'balanced', random_state = RANDOM_STATE),
                    n_jobs = -1,
                    random_state = RANDOM_STATE
                ))
            ]),
            
            'search_spaces': {
                'classifier__n_estimators': Integer(200, 350), 
                'classifier__bootstrap': Categorical([False]),
                'classifier__max_samples': Real(0.4, 0.7, prior='uniform'),

                # Decision tree base estimator hyperparameters 
                'classifier__estimator__max_depth': Integer(3, 8), 
                'classifier__estimator__min_samples_leaf': Integer(80, 150),
                'classifier__estimator__min_samples_split': Integer(160, 300),
                'classifier__estimator__max_features': Categorical(['sqrt', 0.3]), 

                # SMOTE tuning
                'smote__k_neighbors': Integer(3, 5),
                'smote__sampling_strategy': Real(0.8, 1.0, prior='uniform'), 

                # Feature Selection (SelectFromModel) Hyperparameters
                'feature_selection__max_features': Integer(8, 12), 
            }
        },


        'XGBClassifier': {
            'pipeline': ImbPipeline(steps = [
                ('preprocessor', preprocessor),
                ('classifier', XGBClassifier(random_state = RANDOM_STATE, use_label_encoder = False, eval_metric = 'logloss', n_jobs = -1))
            ]),
            'search_spaces': {
                'classifier__n_estimators': Integer(2000, 4000),
                'classifier__learning_rate': Real(0.001, 0.005, prior = 'log-uniform'),
                
                'classifier__max_depth': Integer(2, 3), # extremely shallow trees
                'classifier__min_child_weight': Real(5.0, 20.0, prior = 'log-uniform'),
                'classifier__subsample': Real(0.7, 0.9), 
                'classifier__colsample_bytree': Real(0.7, 0.9), 
                'classifier__gamma': Real(0.1, 0.5, prior = 'uniform'), 
                'classifier__scale_pos_weight': Real(100.0, 300.0, prior = 'uniform') # high to push for recall
            }
        },


        'SVC': {
            'pipeline': ImbPipeline(steps = [
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state = RANDOM_STATE)),
                ('classifier', SVC(probability = True, random_state = RANDOM_STATE))
            ]),
            'search_spaces': {
                'classifier__C': Real(0.01, 10.0, prior = 'log-uniform'), 
                'classifier__kernel': Categorical(['rbf']),
                'classifier__gamma': Real(1e-7, 0.0001, prior = 'log-uniform'),
                'classifier__shrinking': Categorical([True]),
                'smote__k_neighbors': Integer(3, 5), 
                'smote__sampling_strategy': Real(0.3, 0.7, prior = 'uniform')
            }
        },

    }

    return models


def run_training(X_train, y_train, models, custom_cv_generator_func, custom_validation_periods, scorer, n_iter = N_ITER_SEARCH, random_state = RANDOM_STATE):
    """
    Runs hyperparameter training for multiple models using BayesSearchCV.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        models (dict): Dictionary of models, pipelines, and search spaces.
        custom_cv_generator_func (callable): The function to generate CV folds (e.g., custom_cv.generate_custom_cv_folds).
        custom_validation_periods (list): List of dictionaries defining validation periods.
        scorer (callable): Scoring function for BayesSearchCV.
        n_iter (int): Number of search iterations.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: (all_trained_pipelines, model_best_params)
    """
    all_trained_pipelines = {}
    model_best_params = {}

    for model_name, model_info in models.items():
        print(f"\n\n{'='*80}")
        print(f"--- Starting Training & Hyperparameter Tuning for Model: {model_name} ---")
        print(f"{'='*80}\n")

        current_pipeline = model_info['pipeline']
        current_search_spaces = model_info['search_spaces']

        # --- Re-instantiate the CV generator for each model ---
        # This ensures each BayesSearchCV gets a fresh set of folds
        current_custom_cv_folds = custom_cv_generator_func(X_train, y_train, custom_validation_periods)

        if not current_search_spaces:
            print(f"Skipping BayesSearchCV for {model_name}, no search spaces defined. Fitting default pipeline.")
            best_pipeline = current_pipeline
            best_pipeline.fit(X_train, y_train)
            model_best_params[model_name] = {}
        else:
            bayes_search = BayesSearchCV(
                estimator = current_pipeline,
                search_spaces = current_search_spaces,
                cv = current_custom_cv_folds,
                scoring = scorer,
                n_jobs = -1,
                n_iter = n_iter,
                random_state = random_state,
                verbose = 1
            )
            logger.info(f"Fitting BayesSearchCV for {model_name} on X_train (shape: {X_train.shape})...")
            bayes_search.fit(X_train, y_train)

            print(f"\nBest Parameters for {model_name}:")
            print(bayes_search.best_params_)
            best_pipeline = bayes_search.best_estimator_
            model_best_params[model_name] = bayes_search.best_params_

        model_filename = f"models/{model_name}_best_pipeline.pkl"
        joblib.dump(best_pipeline, model_filename)
        print(f"Saved best pipeline for {model_name} to {model_filename}")
        all_trained_pipelines[model_name] = best_pipeline

    return all_trained_pipelines, model_best_params


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_economic_data_stationary.csv", parse_dates = ["date"], index_col = "date")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df, return_preprocessor = True)
    scorer = make_scorer(f1_score, pos_label = 1, zero_division = 0)
    models = initialize_models(preprocessor)
    print("Running model training as standalone script...")
    custom_validation_periods = [
        (datetime(1981, 7, 1), datetime(1982, 10, 31)),
        (datetime(1990, 7, 1), datetime(1991, 2, 28)),
        (datetime(2001, 3, 1), datetime(2001, 10, 31)) 
    ]

    trained_pipelines, best_params = run_training(
        X_train,
        y_train,
        models,
        custom_cv.generate_custom_cv_folds, 
        custom_validation_periods, 
        scorer,
        n_iter = N_ITER_SEARCH,
        random_state = RANDOM_STATE
    )
