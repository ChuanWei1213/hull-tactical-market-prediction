# Disable warnings
import warnings
warnings.filterwarnings('ignore')

# Handle data
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10, 6)

# Model and evaluation
from powershap import PowerShap
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import root_mean_squared_error, accuracy_score, r2_score, confusion_matrix

# Typing
from typing import Literal
from sklearn.base import RegressorMixin, ClassifierMixin

# Progress bar
from tqdm import tqdm

# Introspection
import inspect

# File system
from pathlib import Path

def get_model_type(model: RegressorMixin | ClassifierMixin) -> Literal['classifier', 'regressor']:
    return model._estimator_type

def get_result_path(dataset_name: str, window: int, delta: int, h: int, model: RegressorMixin | ClassifierMixin, category: str) -> Path:
    model_name = model.__class__.__name__
    return Path('results') / dataset_name / f'window={window}&delta={delta}&h={h}' / model_name / f'{category}.csv'

def save_result_csv(dataset_name: str, window: int, delta: int, h: int, model: RegressorMixin | ClassifierMixin, category: str, data: pd.DataFrame | pd.Series):
    path = get_result_path(dataset_name, window, delta, h, model, category)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, header=True)

def get_result_csv(dataset_name: str, window: int, delta: int, h: int, model: RegressorMixin | ClassifierMixin, category: str) -> pd.DataFrame | pd.Series:
    path = get_result_path(dataset_name, window, delta, h, model, category)
    return pd.read_csv(path, index_col='date_id').squeeze() if path.exists() else None

def train(X: pd.DataFrame, y: pd.Series, window: int, delta: int, h: int, model: RegressorMixin | ClassifierMixin, ps_periods: int = 20):
    if 'name' not in X.attrs:
        raise ValueError('X must have a "name" attribute for dataset identification.')
    dataset_name = X.attrs['name']
    if get_result_csv(dataset_name, window, delta, h, model, 'y_true') is not None:
        return
    
    model_type = get_model_type(model)
    model_name = model.__class__.__name__

    # h-day forward returns as target
    y = y.rolling(h).apply(lambda x: (1 + x).prod() - 1, raw=True).shift(-h+1)
    if model_type == 'classifier':
        y = (y > 0).astype(int)
    
    # Total time points
    T = X.shape[0]

    # Number of iterations
    K = int((T - (window + 2 * h)) / delta)

    # Conduct PowerShap every ps_periods
    ps_iter_periods = int(ps_periods / delta)
    
    # Predicitions with all features and PowerShap selected features
    base_pred = pd.Series(index=y.index)
    ps_pred = base_pred.copy()
    
    # Number of selected features over time
    n_feats = pd.Series(index=X.index[window+h:K*delta+window+h:ps_periods])
    
    # Impacts and impact ratios over time
    impacts = pd.DataFrame(index=X.index[window+h:K*delta+window+h:ps_periods], columns=X.columns)
    impacts_ratios = impacts.copy()
    
    # PowerShap selector and selected features
    selector = None
    feats = np.array([])
    
    # Check if model.fit supports verbose parameter
    fit_params = inspect.signature(model.fit).parameters
    fit_kwargs = {'verbose': 0} if 'verbose' in fit_params else {}

    for k in tqdm(range(K), desc=f'Processing {model_name} with window={window}, delta={delta}, h={h}'):
        t = k * delta
        train_indices = slice(t, t + window)
        test_indices = slice(t + window + h, t + window + h + delta)
        X_train = X.iloc[train_indices].dropna(axis=1)
        y_train = y.iloc[train_indices]
        X_test = X.iloc[test_indices][X_train.columns]
        
        # Fit PowerShap periodically
        if selector is None or k % ps_iter_periods == 0:
            selector = PowerShap(
                model = model,
                automatic=True,
                verbose=0,
                show_progress=False
            )
            selector.fit(X_train, y_train, **fit_kwargs)
            
            feats = selector.get_feature_names_out()
            n_feats.loc[X.index[t + window + h]] = len(feats)
            
            impact = selector._processed_shaps_df['impact'].drop(index=['random_uniform_feature'])
            
            impacts.loc[X.index[t + window + h]] = impact
            impacts_ratios.loc[X.index[t + window + h]] = impact / impact.sum()
        
        # Skip if not enough classes for classifier or no features selected
        if model_type == 'classifier' and y_train.nunique() < 2 or len(feats) == 0:
            continue
        
        base_pred[X.index[test_indices]] = model.fit(X_train, y_train, **fit_kwargs).predict(X_test)
        ps_pred[X.index[test_indices]] = model.fit(X_train[feats], y_train, **fit_kwargs).predict(X_test[feats])
            
    # Save results
    for category, values in zip(
        ['y_true', 'base_pred', 'ps_pred', 'impacts', 'impacts_ratios', 'n_feats'],
        [y, base_pred, ps_pred, impacts, impacts_ratios, n_feats]
    ):
        save_result_csv(dataset_name, window, delta, h, model, category, values)
        
def evaluate(dataset_name: str, window: int, delta: int, h: int, model):
    y_true = get_result_csv(dataset_name, window, delta, h, model, 'y_true')
    base_pred = get_result_csv(dataset_name, window, delta, h, model, 'base_pred')
    ps_pred = get_result_csv(dataset_name, window, delta, h, model, 'ps_pred')
    
    mask = ~np.isnan(ps_pred)
    y_true = y_true[mask]
    base_pred = base_pred[mask]
    ps_pred = ps_pred[mask]
    
    metrics = {
        'RMSE': root_mean_squared_error,
        'Accuracy': accuracy_score
    }
    metric = 'RMSE' if get_model_type(model) == 'regressor' else 'Accuracy'
    metric_fn = metrics[metric]
    scores = metric_fn(y_true, base_pred), metric_fn(y_true, ps_pred)
    
    model_name = model.__class__.__name__
    print(f'{model_name} {metric}: {scores[0]:.6f}')
    print(f'{model_name} with PowerShap {metric}: {scores[1]:.6f}')
    print('---')
    
    return scores

def plot_results(dataset_name, window, delta, h, model):
    y_true = get_result_csv(dataset_name, window, delta, h, model, 'y_true')
    base_pred = get_result_csv(dataset_name, window, delta, h, model, 'base_pred')
    ps_pred = get_result_csv(dataset_name, window, delta, h, model, 'ps_pred')
    
    mask = ~np.isnan(ps_pred)
    y_true = y_true[mask]
    base_pred = base_pred[mask]
    ps_pred = ps_pred[mask]
    
    model_name = model.__class__.__name__
    
    if get_model_type(model) == 'regressor':
        # Scatter plot with r2 score
        plt.figure(figsize=(16, 6))
        
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=y_true, y=base_pred)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.title(rf'All Features, $R^2$: {r2_score(y_true, base_pred):.4f}')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=y_true, y=ps_pred)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.title(rf'PowerShap Selected Features, $R^2$: {r2_score(y_true, ps_pred):.4f}')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        
        plt.suptitle(f'{model_name} Predictions vs True Values')
        plt.show()
    else:
        # Confusion matrix
        cm_base = confusion_matrix(y_true, base_pred)
        cm_ps = confusion_matrix(y_true, ps_pred)
        
        plt.figure(figsize=(16, 6))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues')
        plt.title('All Features Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(cm_ps, annot=True, fmt='d', cmap='Blues')
        plt.title('PowerShap Selected Features Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plt.suptitle(f'{model_name} Confusion Matrices')
        plt.show()
    
def plot_feature_impacts(dataset_name, window, delta, h, model):
    n_feats = get_result_csv(dataset_name, window, delta, h, model, 'n_feats')
    impacts = get_result_csv(dataset_name, window, delta, h, model, 'impacts')
    impacts_ratios = get_result_csv(dataset_name, window, delta, h, model, 'impacts_ratios')
    
    avg_impacts = impacts.median().sort_values(ascending=False)[:10]
    avg_impacts_ratios = impacts_ratios.median().sort_values(ascending=False)[:10]
    
    desc = f'model={model.__class__.__name__}, window={window}, delta={delta}, h={h}'
    
    # Number of features over time
    plt.figure(figsize=(16, 6))
    sns.lineplot(data=n_feats, legend=False)
    plt.title(f'Number of Selected Features Over Time ({desc})')
    plt.xlabel(n_feats.index.name)
    plt.ylabel('Number of Features')
    plt.show()
    
    # Number of features box plot
    plt.figure(figsize=(6, 6))
    sns.boxplot(data=n_feats)
    plt.title(f'Number of Selected Features Distribution ({desc})')
    plt.xlabel('Features')
    plt.ylabel('Number of Features')
    plt.show()
    
    # Median impacts bar plot
    plt.figure(figsize=(16, 6))
    sns.barplot(x=avg_impacts.index, y=avg_impacts.values)
    plt.xticks(rotation=90)
    plt.title(f'Median Feature Impacts ({desc})')
    plt.xlabel('Features')
    plt.ylabel('Median Impact')
    plt.show()
    
    # Median impact ratios bar plot
    plt.figure(figsize=(16, 6))
    sns.barplot(x=avg_impacts_ratios.index, y=avg_impacts_ratios.values)
    plt.xticks(rotation=90)
    plt.title(f'Median Feature Impact Ratios ({desc})')
    plt.xlabel('Features')
    plt.ylabel('Median Impact Ratio')
    plt.show()
    
    # Impact over time
    max_impacts = impacts.max()
    top_10_features = max_impacts.nlargest(10).index
    plt.figure(figsize=(15, 8))
    for col in top_10_features:
        line = plt.plot(impacts.index, impacts[col], label=col)
        max_val = impacts[col].max()
        max_idx = impacts[col].idxmax()
        plt.text(max_idx, max_val, col, color=line[0].get_color())
    plt.title(f'Feature Impacts Over Time ({desc})')
    plt.xlabel(impacts.index.name)
    plt.ylabel('Impact')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    
    # Impact ratios over time
    max_ratios = impacts_ratios.max()
    top_10_features_ratios = max_ratios.nlargest(10).index
    plt.figure(figsize=(15, 8))
    for col in top_10_features_ratios:
        line = plt.plot(impacts_ratios.index, impacts_ratios[col], label=col)
        max_val = impacts_ratios[col].max()
        max_idx = impacts_ratios[col].idxmax()
        plt.text(max_idx, max_val, col, color=line[0].get_color())
    plt.title(f'Feature Impact Ratios Over Time ({desc})')
    plt.xlabel(impacts_ratios.index.name)
    plt.ylabel('Impact Ratio')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()