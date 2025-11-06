import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder # To convert text labels to numbers

# --- 1. Load Your Data ---
try:
    data = pd.read_csv("../rdf/master_csv/master_rdf_dataset.csv")
except FileNotFoundError:
    print("Error: Master dataset file not found.")
    exit()

# --- 2. Prepare Data for ML ---

# Separate features (X) from the target/label (y)
X = data.drop(columns=['structure_id', 'label']) # All g(r) columns
y = data['label']                                # The 'label' column

# Convert text labels (e.g., "BCC", "FCC") into numbers (e.g., 0, 1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y
)

print(f"Data loaded: {len(X_train)} training samples, {len(X_test)} testing samples.")

# --- 3. Create and Train the Random Forest Model ---

# Initialize the classifier
# n_estimators = the number of "trees" in the forest. 100 is a good default.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

print("Training the Random Forest model...")
# Train the model on the training data
rf_model.fit(X_train, y_train)

print("Training complete.")

# --- 4. Evaluate the Model ---

# Make predictions on the unseen test data
y_pred = rf_model.predict(X_test)

# Convert numerical predictions back to text labels for the report
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Print a detailed report (precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels))