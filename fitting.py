import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pickle
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import optuna


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred)

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    return precision_score(y_true, y_pred)

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    return recall_score(y_true, y_pred)

def get_metrics(y_train: np.ndarray,
                y_train_pred: np.ndarray,

                y_test: np.ndarray,
                y_pred: np.ndarray,
                name: str,
                model,
                x: np.ndarray,
                y: np.ndarray,
                ):
    """Генерация таблицы с метриками"""
    df_metrics = pd.DataFrame()

    df_metrics['model'] = [name]

    df_metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
    df_metrics['test_accuracy'] = accuracy_score(y_test, y_pred)

    df_metrics['recall'] = recall_score(y_test, y_pred)
    df_metrics['precision'] = precision_score(y_test, y_pred)

    df_metrics['F1'] = f1_score(y_test, y_pred)

    df_metrics['cv_roc_auc'] = cross_val_score(model, x, y, cv=5, scoring='roc_auc').mean()
    df_metrics['pr_auc'] = cross_val_score(model, x, y, cv=5, scoring='average_precision').mean()
    return df_metrics


df = pd.read_csv('data/input/dataset.csv')
users = df['user_id']


# coding target
df.loc[df['full_pnl']>=0, 'full_pnl'] = 0
df.loc[df['full_pnl']<0, 'full_pnl'] = 1


random = 42
x = df.drop(columns={'full_pnl', 'user_id'})
y = np.array(df['full_pnl']).reshape((-1, 1))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=random)


scaler = StandardScaler()
x_train_proc = scaler.fit_transform(x_train)
x_test_proc = scaler.fit_transform(x_test)


def objective(trial):
    class_imbalance_ratio = round(len(df[df['full_pnl'] == 0]) / len(df[df['full_pnl'] == 1]), 2)
    params = {
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart', 'gbtree', 'dart']),
        'learning_rate': trial.suggest_float('learning_rate', 0, 1),
        'max_depth': trial.suggest_int('max_depth', 4, 20),
        'min_split_loss': trial.suggest_int('min_split_loss', 0, 700),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 3000),
        'max_delta_step': trial.suggest_float('max_delta_step', 0, 20), #дизсбаланс в классах
        'subsample': trial.suggest_float('subsample', 0.05, 1),
        'tree_method': trial.suggest_categorical('tree_method', ['auto', 'exact', 'approx']),
        'scale_pos_weight': class_imbalance_ratio,
        'n_jobs': 3
    }
    xgb = XGBClassifier(**params).fit(x_train_proc, y_train)
    score = cross_val_score(X=x_train_proc, y=y_train, scoring='f1_macro', cv=3, estimator=xgb).mean()
    return score


study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective, n_trials=10000)


xgb = XGBClassifier(**study_xgb.best_params).fit(x_train_proc, y_train)
metrics_df = get_metrics(y_train, xgb.predict(x_train_proc), y_test, xgb.predict(x_test_proc), 'xgb', xgb, x, y)
print(metrics_df[['test_accuracy', 'precision', 'F1']])


with open('saved_models/XGBClassifier_f1_tune.pickle', 'wb') as file:
    pickle.dump(xgb, file)


feature_importance = xgb.get_booster().get_fscore()
feature_importance = [feature_importance[f'f{i}'] for i in range(len(feature_importance))]
feature_importance = pd.DataFrame({
    'feature': x.columns,
    'value': feature_importance
})
print(feature_importance.sort_values('value'))