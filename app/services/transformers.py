import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import BaseEstimator
from datetime import datetime

# -------------------------
# Utility functions placeholders
# -------------------------
def format_mine_plan_dates(X): return X
def format_shipping_plan_dates(X): return X
def format_weather_dates(X): return X
def format_hauling_dates(X): return X
def encode_priority_flag(X, col): 
    X = X.copy()
    mapping = {"High": 3, "Medium": 2, "Low": 1}
    if col in X:
        X[col + "_score"] = X[col].map(mapping)
    return X
def encode_weather_category(X): return X

# -------------------------
# CLASS DEFINITIONS
# -------------------------

class DateFormatter(BaseEstimator, TransformerMixin):
    def __init__(self, date_columns, dataset_type='mine'):
        self.date_columns = date_columns
        self.dataset_type = dataset_type
    
    def fit(self, X, y=None): return self
    
    def transform(self, X):
        X = X.copy()
        if self.dataset_type == 'mine':
            return format_mine_plan_dates(X)
        elif self.dataset_type == 'shipping':
            return format_shipping_plan_dates(X)
        elif self.dataset_type == 'weather':
            return format_weather_dates(X)
        elif self.dataset_type == 'hauling':
            return format_hauling_dates(X)
        return X

class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean', fill_value=None, columns=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.columns = columns
        self.fill_values_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        if self.columns is None:
            self.columns = X.columns[X.isnull().any()].tolist()
        for col in self.columns:
            if self.strategy == 'mean' and X[col].dtype != 'object':
                self.fill_values_[col] = X[col].mean()
            elif self.strategy == 'median' and X[col].dtype != 'object':
                self.fill_values_[col] = X[col].median()
            elif self.strategy == 'mode':
                self.fill_values_[col] = X[col].mode()[0]
            else:
                self.fill_values_[col] = self.fill_value
        return self

    def transform(self, X):
        X = X.copy()
        for col, val in self.fill_values_.items():
            if col in X.columns:
                X[col].fillna(val, inplace=True)
        return X

class PriorityEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, priority_column='priority_flag'):
        self.priority_column = priority_column

    def fit(self, X, y=None): return self

    def transform(self, X):
        return encode_priority_flag(X, self.priority_column)

class WeatherCategoryEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X):
        return encode_weather_category(X)

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, method='iqr', multiplier=1.5):
        self.columns = columns
        self.method = method
        self.multiplier = multiplier
        self.bounds_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        if self.columns is None:
            self.columns = X.select_dtypes(include=np.number).columns.tolist()
        for col in self.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds_[col] = {
                'lower': Q1 - self.multiplier * IQR,
                'upper': Q3 + self.multiplier * IQR
            }
        return self

    def transform(self, X):
        X = X.copy()
        for col, b in self.bounds_.items():
            X = X[(X[col] >= b['lower']) & (X[col] <= b['upper'])]
        return X

class DuplicateRemover(BaseEstimator, TransformerMixin):
    def __init__(self, subset=None):
        self.subset = subset

    def fit(self, X, y=None): return self

    def transform(self, X):
        return X.copy().drop_duplicates(subset=self.subset)

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None): return self

    def transform(self, X):
        X = X.copy()
        for col in self.feature_names:
            if col not in X:
                X[col] = 0
        return X[self.feature_names].fillna(X[self.feature_names].mean())

class PreprocessorMining(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.priority_map = {"High": 3, "Medium": 2, "Low": 1}

    def fit(self, X, y=None): return self

    def transform(self, X):
        X = X.copy()
        if 'plan_date' in X.columns:
            X['plan_date'] = pd.to_datetime(X['plan_date'])
        if 'priority_flag' in X.columns:
            X['priority_score'] = X['priority_flag'].map(self.priority_map)
        if 'precipitation_mm' in X.columns:
            def weather(row):
                if row > 20: return "Heavy Rain"
                if row > 5: return "Moderate Rain"
                if row > 0: return "Light Rain"
                return "Clear"
            X['weather_category'] = X['precipitation_mm'].apply(weather)
        return X

class PreprocessorShipping(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if 'ETA' in X:
            X['ETA'] = pd.to_datetime(X['ETA'])
            X['eta_date'] = pd.to_datetime(X['ETA'].dt.date)
        if 'ETD' in X:
            X['ETD'] = pd.to_datetime(X['ETD'])
        if 'ETA' in X and 'ETD' in X:
            X['loading_duration_hr'] = (X['ETD'] - X['ETA']).dt.total_seconds() / 3600
        return X

class MiningPipelineWrapper(BaseEstimator):
    def __init__(self, mining_model, priority_pipeline):
        self.mining_model = mining_model
        self.priority_pipeline = priority_pipeline
        self.preprocessor = PreprocessorMining()

    def fit(self, X, y=None): return self

    def predict(self, X):
        Xp = self.preprocessor.transform(X)
        Xp['ai_priority_score'] = self.priority_pipeline.predict(X)
        features = [
            "planned_production_ton",
            "hauling_distance_km",
            "ai_priority_score",
            "precipitation_mm",
            "wind_speed_kmh",
            "cloud_cover_pct",
            "temp_day"
        ]
        Xs = Xp[features].fillna(Xp[features].mean())
        return self.mining_model.predict(Xs)
