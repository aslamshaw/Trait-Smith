import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from category_encoders import LeaveOneOutEncoder
from sklearn.feature_selection import mutual_info_classif


class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy_num='mean', strategy_cat='most_frequent'):
        self.strategy_num = strategy_num
        self.strategy_cat = strategy_cat
        self.num_cols_ = None
        self.cat_cols_ = None
        self.num_imputer_ = None
        self.cat_imputer_ = None

    def fit(self, X, y=None):
        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()

        self.num_imputer_ = SimpleImputer(strategy=self.strategy_num)
        self.cat_imputer_ = SimpleImputer(strategy=self.strategy_cat)

        self.num_imputer_.fit(X[self.num_cols_])
        self.cat_imputer_.fit(X[self.cat_cols_])
        return self

    def transform(self, X):
        X_ = X.copy()
        X_[self.num_cols_] = self.num_imputer_.transform(X_[self.num_cols_])
        X_[self.cat_cols_] = self.cat_imputer_.transform(X_[self.cat_cols_])
        return X_


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bin_edges_ = None

    def fit(self, X, y=None):
        self.bin_edges_ = pd.qcut(X['Time_spent_Alone'], q=3, retbins=True, duplicates='drop')[1]
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['Alone_to_Social_Ratio'] = X_['Time_spent_Alone'] / (X_['Social_event_attendance'] + 1)
        X_['Social_Activity_Index'] = X_['Friends_circle_size'] + X_['Post_frequency'] + X_['Going_outside']
        X_['Time_spent_Alone_Binned'] = pd.cut(X_['Time_spent_Alone'],
                                               bins=self.bin_edges_,
                                               labels=['Low', 'Medium', 'High'],
                                               include_lowest=True)
        return X_


class LOOEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, min_unique=10):
        self.min_unique = min_unique
        self.encoder_ = None
        self.columns_ = None

    def fit(self, X, y):
        cat_cols_ = X.select_dtypes(include=['object', 'category']).columns
        self.columns_ = [col for col in cat_cols_ if X[col].nunique() >= self.min_unique]
        if self.columns_:
            self.encoder_ = LeaveOneOutEncoder(
                cols=self.columns_,
                handle_unknown='value',
                handle_missing='value'
            )
            self.encoder_.fit(X[self.columns_], y)
        return self

    def transform(self, X):
        X_ = X.copy()
        if self.columns_:
            X_[self.columns_] = self.encoder_.transform(X_[self.columns_])
        return X_


class MIFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self.selected_features_ = None

    def fit(self, X, y):
        num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_data = X[num_cols_]
        cat_data = X[cat_cols_]

        encoder_ = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        cat_encoded_np = encoder_.fit_transform(cat_data)
        cat_encoded = pd.DataFrame(cat_encoded_np, columns=cat_cols_, index=X.index)

        X_mi_df = pd.concat([num_data, cat_encoded], axis=1)
        discrete_features = [True if col in cat_cols_ else False for col in X_mi_df.columns]

        mi_scores = mutual_info_classif(X_mi_df, y, discrete_features=discrete_features)
        mi_series = pd.Series(mi_scores, index=X_mi_df.columns)
        selected_features = mi_series[mi_series >= self.threshold].index.tolist()

        self.selected_features_ = selected_features
        return self

    def transform(self, X):
        return X[self.selected_features_]


class AutoScaleEncode(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pipeline_ = None
        self.all_cols = None

    def fit(self, X, y=None):
        num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.pipeline_ = ColumnTransformer([
            ('scale_num', StandardScaler(), num_cols_),
            ('encode_cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             cat_cols_)])
        self.pipeline_.fit(X, y)

        cat_encoder = self.pipeline_.named_transformers_['encode_cat']
        encoded_col_names = cat_encoder.get_feature_names_out(cat_cols_).tolist()
        self.all_cols = num_cols_ + encoded_col_names
        return self

    def transform(self, X):
        X_final = pd.DataFrame(self.pipeline_.transform(X), columns=self.all_cols, index=X.index)
        return X_final
