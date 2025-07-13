import os
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from ml_pipeline import *


# Load dataset
data_path = os.path.join(os.path.dirname(__file__), 'personality_dataset.csv')
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}. Please check the file path.")
data = pd.read_csv(data_path)
target_col = 'Personality'
data = data.dropna(subset=[target_col])

# Train-test split
X = data.drop(columns=target_col)
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Keep only rows in test where target label is known in training
valid_classes = set(np.unique(y_train))
mask = y_test.isin(valid_classes)
X_test = X_test[mask]
y_test = y_test[mask]

# Target label encoding
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Pipelines per type
preprocessing = Pipeline([('imputation', DataFrameImputer()),
                          ('feature_engineering', FeatureEngineer()),
                          ('target_encoding', LOOEncoder()),
                          ('feature_selection', MIFeatureSelector()),
                          ('scale_encode', AutoScaleEncode())])

# Define models with expanded hyperparameter grids
model_pipelines = {
    'rf': {
        'pipeline': Pipeline(
            [('preprocessing', preprocessing), ('randomforestclassifier', RandomForestClassifier(random_state=42))]),
        'params': {
            'preprocessing__feature_selection__threshold': [0.0, 0.001, 0.01, 0.05, 0.1],
            'randomforestclassifier__n_estimators': [100, 150],
            'randomforestclassifier__max_depth': [None, 10],
            'randomforestclassifier__min_samples_split': [2, 5]
        }
    },
    'gb': {
        'pipeline': Pipeline([('preprocessing', preprocessing),
                              ('gradientboostingclassifier', GradientBoostingClassifier(random_state=42))]),
        'params': {
            'preprocessing__feature_selection__threshold': [0.0, 0.001, 0.01, 0.05, 0.1],
            'gradientboostingclassifier__n_estimators': [100, 200],
            'gradientboostingclassifier__learning_rate': [0.05, 0.1],
            'gradientboostingclassifier__max_depth': [3, 5]
        }
    },
    'xgb': {
        'pipeline': Pipeline([('preprocessing', preprocessing),
                              ('xgbclassifier', XGBClassifier(random_state=42, eval_metric='logloss'))]),
        'params': {
            'preprocessing__feature_selection__threshold': [0.0, 0.001, 0.01, 0.05, 0.1],
            'xgbclassifier__n_estimators': [100, 150],
            'xgbclassifier__learning_rate': [0.05, 0.1],
            'xgbclassifier__max_depth': [3, 5]
        }
    },
    'svm': {
        'pipeline': Pipeline([('preprocessing', preprocessing), ('svc', SVC(probability=True))]),
        'params': {
            'preprocessing__feature_selection__threshold': [0.0, 0.001, 0.01, 0.05, 0.1],
            'svc__C': np.logspace(-3, 3, 5),
            'svc__kernel': ['rbf', 'linear']
        }
    }
}

# Hyperparameter tuning
best_models = {}

for name, mp in model_pipelines.items():
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    clf = RandomizedSearchCV(mp['pipeline'], mp['params'], n_iter=2, cv=skf, scoring='f1_weighted', n_jobs=-1)
    clf.fit(X_train, y_train)
    best_models[name] = clf.best_estimator_
    print(f"{name} best score: {clf.best_score_:.3f}")

# Stacking optimized models
stacking_model = StackingClassifier(
    estimators=[(name, model) for name, model in best_models.items()],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    n_jobs=-1
)
stacking_model.fit(X_train, y_train)

# Pickle model and label encoder
with open("stacking_model.pkl", "wb") as f:
    pickle.dump(stacking_model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
