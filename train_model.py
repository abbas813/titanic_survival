import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Titanic dataset
df = pd.read_csv("Titanic_train.csv")

# Select relevant features and drop missing values
df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].dropna()

# Convert categorical variable 'Sex' to numerical
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Split features and target
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save the trained model
with open("logistic_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save the scaler
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# Evaluate model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained and saved successfully! Accuracy: {accuracy:.2f}")
