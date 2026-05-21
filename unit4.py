#Importing the data
from pathlib import Path
import pandas as pd


pa = Path(__file__).parent / "resources" / "unit_4" / "data"
births = pd.read_csv(pa / "1971.csv.gz")

births_dropped = births[['momage', 'dadage', 'plurality', 'birthorder', 'birthweight']].dropna()
print(births_dropped.head())

# Spliting the training and test data
from sklearn.model_selection import train_test_split
X = births_dropped[['momage', 'dadage', 'plurality', 'birthorder']]
y = births_dropped['birthweight']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=23
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_test, test_pred)

print(f"Training MSE: {train_mse:.4f}")
print(f"Test MSE:     {test_mse:.4f}")


