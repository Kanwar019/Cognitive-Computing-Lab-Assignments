from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load and Prepare the Data
iris = load_iris()
X = iris.data
y = iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=102317223)

# 2. Create and Train the Model
logistic_model = LogisticRegression(solver='liblinear', multi_class='ovr')  # 'liblinear' for small datasets
logistic_model.fit(X_train, y_train)

# 3. Make Predictions
y_pred = logistic_model.predict(X_test)

# 4. Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print(classification_report(y_test, y_pred))
