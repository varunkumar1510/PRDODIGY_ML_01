import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the train and test data
train_df = pd.read_csv('train.csv')  # Replace with your train dataset path
test_df = pd.read_csv('test.csv')    # Replace with your test dataset path

# Select relevant features
features = ['GrLivArea', 'Bedroom', 'FullBath', 'HalfBath', 'TotRmsAbvGrd']
target = 'SalePrice'

# Prepare the data for modeling
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]

# Create a preprocessing pipeline
numeric_features = ['GrLivArea', 'Bedroom', 'FullBath', 'HalfBath', 'TotRmsAbvGrd']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Create a pipeline that includes preprocessing and regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model on training data
mae_train = mean_absolute_error(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
r2_train = r2_score(y_train, y_pred_train)

print('Training set evaluation:')
print(f'MAE: {mae_train}')
print(f'MSE: {mse_train}')
print(f'RMSE: {rmse_train}')
print(f'R^2: {r2_train}')

# Since we don't have true SalePrice for the test data, we won't evaluate it here
print('Predictions for the test set:')
print(y_pred_test)
