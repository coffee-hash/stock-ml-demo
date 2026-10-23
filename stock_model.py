import yfinance as yf
import warnings
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np


warnings.filterwarnings('ignore')
df = yf.download("RELIANCE.NS", start="2015-01-01", end="2026-10-23")


df.columns = df.columns.droplevel(1)
df = df.dropna()
df['return_1d'] = df['Close'].pct_change(1)
df['return_2d'] = df['Close'].pct_change(2)
df['return_5d'] = df['Close'].pct_change(5)
df['ma_5'] = df['Close'].rolling(window=5).mean()
df['ma_20'] = df['Close'].rolling(window=20).mean()
df['volume_change']=df['Volume'].pct_change(1)

df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.dropna()
df = df.replace([np.inf, -np.inf], np.nan).dropna()

lr_model = LogisticRegression()
rf_model = RandomForestClassifier(random_state=42)

# 1. Define features (X) and target (y)
features = ['return_1d', 'return_2d', 'return_5d', 'ma_5', 'ma_20', 'volume_change']
X = df[features]
y = df['target']

split_index = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]


# baseline = DummyClassifier(strategy='most_frequent')
# baseline.fit(X_train,y_train)
# baseline_preds = baseline.predict(X_test)

lr_model.fit(X_train,y_train)
rf_model.fit(X_train,y_train)
lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)


# 2. Print Results for ALL models

joblib.dump(rf_model, 'rf_stock_model.pkl')
# print("\n--- Logistic Regression ---")
# print("Accuracy:", accuracy_score(y_test, lr_preds))
# print("Classification Report:\n", classification_report(y_test, lr_preds))

# print("\n--- Random Forest ---")
# print("Accuracy:", accuracy_score(y_test, rf_preds))
# print("Classification Report:\n", classification_report(y_test, rf_preds))



