import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_diabetes
from sklearn.ensemble import OOBRandomForestRegressor
from sklearn.model_selection import train_test_split

SEED = 14208

diabetes = load_diabetes(as_frame=True)
diabetes = diabetes.frame
train_df, validation_df = train_test_split(diabetes, test_size=0.95, random_state=SEED)
y_train, y_valid = train_df['target'], validation_df['target']
X_train, X_valid = train_df.drop('target', axis=1), validation_df.drop('target', axis=1)

print(X_train)

oob_rf = OOBRandomForestRegressor(random_state=SEED, n_estimators=10)
oob_rf.fit(X_train, y_train)
predictions_def = oob_rf.predict(X_valid)