#################################################
# Principal Component Analysis (PCA) Code Template
#################################################

# Import Required Packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Required for cumulative sum

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------------------------------------------------
# Import Sample Data
data_for_model = pd.read_csv("sample_data_pca.csv")

# -------------------------------------------------------------------
# Drop Unnecessary Columns
data_for_model.drop("user_id", axis=1, inplace=True)

# -------------------------------------------------------------------
# Shuffle the Data
data_for_model = shuffle(data_for_model, random_state=42)

# Check class balance
print(data_for_model["purchased_album"].value_counts(normalize=True))

# -------------------------------------------------------------------
# Deal with Missing Data
print("Missing values in dataset:", data_for_model.isna().sum().sum())
data_for_model.dropna(how="any", inplace=True)

# -------------------------------------------------------------------
# Split Input and Output Variables
X = data_for_model.drop(["purchased_album"], axis=1)
y = data_for_model["purchased_album"]

# -------------------------------------------------------------------
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------------------------
# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------------
# Apply PCA (Full)
pca_full = PCA(n_components=None, random_state=42)
pca_full.fit(X_train_scaled)

explained_variance = pca_full.explained_variance_ratio_
explained_variance_cumulative = np.cumsum(explained_variance)

# -------------------------------------------------------------------
# Plot Variance Explained and Cumulative Variance
num_components = len(explained_variance)
num_vars_list = list(range(1, num_components + 1))

plt.figure(figsize=(15, 10))

# Plot individual component variance
plt.subplot(2, 1, 1)
plt.bar(num_vars_list, explained_variance)
plt.title("Variance across Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("Proportion of Variance Explained")

# Plot cumulative variance
plt.subplot(2, 1, 2)
plt.plot(num_vars_list, explained_variance_cumulative, marker='o')
plt.title("Cumulative Variance across Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Proportion of Variance Explained")

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# Apply PCA to retain 75% variance
pca = PCA(n_components=0.75, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Selected number of components: {pca.n_components_}")

# -------------------------------------------------------------------
# Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_pca, y_train)

# -------------------------------------------------------------------
# Model Accuracy
y_pred_class = clf.predict(X_test_pca)
acc = accuracy_score(y_test, y_pred_class)
print(f"Model Accuracy: {acc:.4f}")
