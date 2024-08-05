import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Load the train and test data
train_df = pd.read_csv('/content/drive/MyDrive/Colab Datasets/train.csv')
test_df = pd.read_csv('/content/drive/MyDrive/Colab Datasets/test.csv')

# Print the column names to check for correct feature names
print(train_df.columns)

# Update relevant features with the correct column names
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'TotRmsAbvGrd']
target = 'SalePrice'

# Prepare the data for modeling
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]

# Create a preprocessing pipeline
numeric_features = features
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

# Create a DataFrame for the evaluation metrics
evaluation_metrics = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'R^2'],
    'Value': [mae_train, mse_train, rmse_train, r2_train]
})

# Create a DataFrame for the train predictions
train_predictions = pd.DataFrame({
    'Id': train_df['Id'],
    'Actual SalePrice': y_train,
    'Predicted SalePrice': y_pred_train
})

# Create a DataFrame for the test predictions
test_predictions = pd.DataFrame({
    'Id': test_df['Id'],
    'Predicted SalePrice': y_pred_test
})

# Save the predictions to CSV files
train_predictions.to_csv('/content/drive/MyDrive/Colab Datasets/train_predictions.csv', index=False)
test_predictions.to_csv('/content/drive/MyDrive/Colab Datasets/test_predictions.csv', index=False)

# Display the evaluation metrics
print("Training set evaluation metrics:")
print(evaluation_metrics)

# Display the train predictions
print("\nTrain set predictions:")
print(train_predictions.head())

# Display the test predictions
print("\nTest set predictions:")
print(test_predictions.head())
