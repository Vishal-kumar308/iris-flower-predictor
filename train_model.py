from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save the model to a file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
