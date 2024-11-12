import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import IQRRandomForestRegressor, RandomForestRegressor, PercentileTrimmingRandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


TEST_IQR = True
TEST_PERCENTILE = True
INCLUDE_DEFAULT = True

SEED = 14208

diabetes = load_diabetes(as_frame=True)
diabetes = diabetes.frame
train_df, validation_df = train_test_split(diabetes, test_size=0.2, random_state=SEED)
y_train, y_valid = train_df['target'], validation_df['target']
X_train, X_valid = train_df.drop('target', axis=1), validation_df.drop('target', axis=1)

def_rf = RandomForestRegressor(random_state=SEED, n_estimators=10)
def_rf.fit(X_train, y_train)
predictions_def = def_rf.predict(X_valid)

if TEST_IQR:
    print("---------------\nRF IQR\n---------------")

    iqr_rf = IQRRandomForestRegressor(random_state=SEED, n_estimators=10, group_size=5)
    iqr_rf.fit(X_train, y_train)
    predictions_iqr = iqr_rf.predict(X_valid)

if TEST_PERCENTILE:
    print("---------------\nRF Percentile\n---------------")

    prc_rf = PercentileTrimmingRandomForestRegressor(random_state=SEED, n_estimators=10, group_size=5, percentile=10)
    prc_rf.fit(X_train, y_train)
    predictions_prc = prc_rf.predict(X_valid)

if INCLUDE_DEFAULT:
    print("\n---------------\nRF Default\n---------------")
    for j in [0,5]:
        preds = []
        group_mean = []
        for i in range(j, 5+j):
            preds.append(def_rf.estimators_[i].predict(X_valid))

        print(f"Grupo {j // 5}\n---------------")
        print("Group preds:", preds)
        print("Group mean:", np.mean(preds, axis=0))
        print('---------------')


if TEST_IQR:
    print('\n---------------')
    print("Predicciones IQR:\n", predictions_iqr)
    mse = mean_squared_error(y_valid, predictions_iqr)
    print(f"MSE {mse}")
    print('---------------')

if TEST_PERCENTILE:
    print('\n---------------')
    print("Predicciones Percentile:\n", predictions_prc)
    mse = mean_squared_error(y_valid, predictions_prc)
    print(f"MSE {mse}")
    print('---------------')

if INCLUDE_DEFAULT:
    print('\n---------------')
    print("Predicciones RF:\n", predictions_def)
    mse = mean_squared_error(y_valid, predictions_def)
    print(f"MSE {mse}")
    print('---------------')
