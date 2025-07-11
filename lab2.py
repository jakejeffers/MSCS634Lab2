# %% [markdown]
# # KNN vs RNN Classifier on Wine Dataset  
# **Name:** Jacob Jeffers  
# **Course:** MSCS 634  
# **Lab:** Lab 2 – KNN and RNN Performance Exploration  
# 

# %%
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the Wine dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

# View feature and class info
print(X.head())
print("\nClass distribution:\n", pd.Series(y).value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k_values = [1, 5, 11, 15, 21]
knn_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    knn_accuracies.append(acc)
    print(f"K = {k} → Accuracy: {acc:.4f}")


# %%
from sklearn.neighbors import RadiusNeighborsClassifier

r_values = [350, 400, 450, 500, 550, 600]
rnn_accuracies = []

for radius in r_values:
    rnn = RadiusNeighborsClassifier(radius=radius, outlier_label=-1)
    rnn.fit(X_train, y_train)
    y_pred = rnn.predict(X_test)

    # Remove outliers (-1 predictions) for accuracy scoring
    filtered = [(pred, actual) for pred, actual in zip(y_pred, y_test) if pred != -1]
    if filtered:
        preds, actuals = zip(*filtered)
        acc = accuracy_score(actuals, preds)
    else:
        acc = 0
    rnn_accuracies.append(acc)
    print(f"Radius = {radius} → Accuracy: {acc:.4f}")


# %%
import matplotlib.pyplot as plt

# Plot KNN accuracy
plt.figure()
plt.plot(k_values, knn_accuracies, marker='o')
plt.title('KNN Accuracy vs k')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# Plot RNN accuracy
plt.figure()
plt.plot(r_values, rnn_accuracies, marker='o', color='green')
plt.title('RNN Accuracy vs Radius')
plt.xlabel('Radius')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()



