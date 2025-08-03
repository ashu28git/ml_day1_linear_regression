import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Load data
df = pd.read_csv('study_hours.csv')
X = df[['Hours']]  # feature
y = df['Scores']   # label

# Step 2: Train the model
model = LinearRegression()
model.fit(X, y)

# Step 3: Predict
predicted = model.predict(X)

# Step 4: Plot
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, predicted, color='red', label='Predicted Line')
plt.title('Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.legend()
plt.show()

# Step 5: Predict new value
hours = float(input("Enter study hours: "))
score = model.predict([[hours]])
print(f"Predicted Score: {score[0]:.2f}")
