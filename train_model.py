import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def data_split():
    df = pd.read_csv('dataset.csv')

    # Drop columns with all NaNs
    df = df.dropna(axis=1, how='all')

    # Handle missing values in features
    X = df.drop('class', axis=1)  # features
    y = df['class']  # target values

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

    return x_train, x_test, y_train, y_test

def train_model():
    x_train, x_test, y_train, y_test = data_split()

    pipelines = {
        'lr': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression())
        ]),
        'rc': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', RidgeClassifier())
        ]),
        'rf': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ]),
        'gb': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier())
        ]),
    }

    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(x_train, y_train)
        fit_models[algo] = model

        # Predict and calculate accuracy
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f'{algo} Accuracy: {accuracy:.4f}')

    return fit_models, x_test, y_test
