
import pandas as pd

df = pd.read_csv("credit_default.csv")

df.head()



from sklearn.model_selection import train_test_split

X = df.drop("DEFAULT_NEXT_MONTH", axis=1)
y = df["DEFAULT_NEXT_MONTH"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Model 1
model1 = RandomForestClassifier(n_estimators=100, class_weight=None, random_state=42)

model1.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)

# Metrics
print("Model 1 Results:")
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Precision:", precision_score(y_test, y_pred1))
print("Recall:", recall_score(y_test, y_pred1))
print("F1 Score:", f1_score(y_test, y_pred1))



# Model 2
model2 = RandomForestClassifier(n_estimators=200, class_weight=None, random_state=42)

# Train it
model2.fit(X_train, y_train)

# Make predictions
y_pred2 = model2.predict(X_test)

# Print results
print("\nModel 2 Results:")
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Precision:", precision_score(y_test, y_pred2))
print("Recall:", recall_score(y_test, y_pred2))
print("F1 Score:", f1_score(y_test, y_pred2))





# Model 3
model3 = RandomForestClassifier(n_estimators=300, class_weight=None, random_state=42)

# Train it
model3.fit(X_train, y_train)

# Make predictions
y_pred3 = model3.predict(X_test)

# Print results
print("\nModel 3 Results:")
print("Accuracy:", accuracy_score(y_test, y_pred3))
print("Precision:", precision_score(y_test, y_pred3))
print("Recall:", recall_score(y_test, y_pred3))
print("F1 Score:", f1_score(y_test, y_pred3))





# Model 4
model4 = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)

model4.fit(X_train, y_train)

y_pred4 = model4.predict(X_test)

print("\nModel 4 Results:")
print("Accuracy:", accuracy_score(y_test, y_pred4))
print("Precision:", precision_score(y_test, y_pred4))
print("Recall:", recall_score(y_test, y_pred4))
print("F1 Score:", f1_score(y_test, y_pred4))





# Model 5
model5 = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)

model5.fit(X_train, y_train)

y_pred5 = model5.predict(X_test)

print("\nModel 5 Results:")
print("Accuracy:", accuracy_score(y_test, y_pred5))
print("Precision:", precision_score(y_test, y_pred5))
print("Recall:", recall_score(y_test, y_pred5))
print("F1 Score:", f1_score(y_test, y_pred5))





# Model 6
model6 = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)

model6.fit(X_train, y_train)

y_pred6 = model6.predict(X_test)

print("\nModel 6 Results:")
print("Accuracy:", accuracy_score(y_test, y_pred6))
print("Precision:", precision_score(y_test, y_pred6))
print("Recall:", recall_score(y_test, y_pred6))
print("F1 Score:", f1_score(y_test, y_pred6))




# Majority Baseline
y_pred_baseline = [0] * len(y_test)

print("\nBaseline Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_baseline))
print("Precision:", precision_score(y_test, y_pred_baseline, zero_division=0))
print("Recall:", recall_score(y_test, y_pred_baseline, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_baseline, zero_division=0))







