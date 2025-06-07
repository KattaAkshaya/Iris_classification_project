
# 📦 Step 1: Import Required Libraries
import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 🎯 Step 2: Extract ZIP File
zip_path = "archive.zip"
extract_folder = "iris_dataset"

if not os.path.exists(extract_folder):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)
    print("✅ Dataset extracted successfully.")
else:
    print("✅ Dataset already extracted.")

# 📄 Step 3: Load the CSV File
csv_path = os.path.join(extract_folder, "Iris.csv")  # Adjust name if needed
df = pd.read_csv(csv_path)

# 👁️ Step 4: Initial Data Exploration
print("\n🔍 First 5 Rows:")
print(df.head())

print("\n📊 Dataset Info:")
print(df.info())

print("\n📈 Target Class Distribution:")
print(df["species"].value_counts())

# 🧹 Step 5: Clean the Dataset
if 'Id' in df.columns:
    df.drop(columns=['Id'], inplace=True)

# 🎯 Step 6: Feature-Target Split
X = df.drop("species", axis=1)
y = df["species"]

# ✂️ Step 7: Split into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌳 Step 8: Train Decision Tree Model
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# 🤖 Step 9: Train Logistic Regression Model
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)

# 🧪 Step 10: Evaluate Models
def evaluate_model(name, model):
    print(f"\n🔍 Evaluation for: {name}")
    y_pred = model.predict(X_test)
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("📋 Classification Report:\n", classification_report(y_test, y_pred))
    print("📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

evaluate_model("Decision Tree", tree_model)
evaluate_model("Logistic Regression", log_model)

# 🔮 Step 11: Make Prediction on Sample Data
sample = [[5.1, 3.5, 1.4, 0.2]]  # Sepal Length, Sepal Width, Petal Length, Petal Width
pred_tree = tree_model.predict(sample)
pred_log = log_model.predict(sample)

print("\n🌼 Sample Prediction (Decision Tree):", pred_tree[0])
print("🌼 Sample Prediction (Logistic Regression):", pred_log[0])

# 🌲 Step 12: Visualize the Decision Tree
plt.figure(figsize=(10, 6))
plot_tree(tree_model, filled=True, feature_names=X.columns, class_names=tree_model.classes_)
plt.title("🌳 Decision Tree Visualization")
plt.show()

# 📉 Step 13: Pairplot for Iris Data
sns.pairplot(df, hue="Species", height=2.5)
plt.suptitle("📊 Pairwise Feature Relationships", y=1.02)
plt.show()
