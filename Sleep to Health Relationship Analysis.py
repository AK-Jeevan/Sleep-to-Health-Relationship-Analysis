# Sleep to Health Relationship Analysis
# This script analyzes the relationship between sleep duration and health outcomes i.e Heart Rate.
# using a dataset that includes various health metrics.
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the dataset
data = pd.read_csv('C:/Users/akjee/Documents/ML/Sleep_health_dataset.csv')

# Remove duplicate rows
data = data.drop_duplicates()

# Remove rows with any missing values
data = data.dropna()

# Display the first few rows of the dataset
print(data.head())

# Display the columns of the dataset
# print(data.columns)

# Display the shape of the dataset i.e number of rows and columns
# print(data.shape)

# Display the statistics of the dataset
print(data.describe())

# To know number of NULL values in each column
# print(data.isnull().sum())
# print(data.info())

# Now define the independent and dependent variables for Multiple Regression
x = pd.DataFrame(data[['Sleep Duration', 'Age', 'Stress Level', 'Physical Activity Level', 'Daily Steps']]) # or use x = pd.DataFrame(data.iloc[:, :-1]) i.e select all rows and all columns except the last one
y = pd.DataFrame(data[['Heart Rate']]) # or use y = pd.DataFrame(data.iloc[:, -1]) i.e select all rows and the last column

# Now divide the dataset into Training set and Testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print(f"X Train shape is :{x_train.shape}")
print(f"X Test shape is :{x_test.shape}")
print(f"Y Train shape is :{y_train.shape}")
print(f"Y Test shape is :{y_test.shape}")

# Now creating the model and fitting it to find patterns
model = LinearRegression()
model.fit(x_train, y_train)

# now to get slope and intercept values for the Independent variables
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Predictions
y_pred = model.predict(x_test)
y_pred = pd.DataFrame(y_pred, columns=['Heart Rate'])
print(y_pred.size)

# to check the accuracy of the model we use metrics
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("Rsquared :", metrics.r2_score(y_test, y_pred))

# Plotting Actual vs Predicted Heart Rate
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.title('Actual vs Predicted Heart Rate')
plt.xlabel('Actual Heart Rate')
plt.ylabel('Predicted Heart Rate')
plt.grid(True)
plt.show()
