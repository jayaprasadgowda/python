import numpy as np
import pandas as pd

# Function to calculate Entropy of a dataset
def entropy(data):
    class_counts = data.iloc[:, -1].value_counts()
    probabilities = class_counts / len(data)
    return -np.sum(probabilities * np.log2(probabilities))

# Function to calculate Information Gain for a feature
def information_gain(data, feature):
    total_entropy = entropy(data)  # Entropy before splitting
    feature_values = data[feature].unique()
    weighted_entropy = 0

    for value in feature_values:
        subset = data[data[feature] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)

    return total_entropy - weighted_entropy  # Information Gain

# Function to select the best feature based on Information Gain
def best_feature(data):
    features = data.columns[:-1]  # Exclude the target class column
    gains = {feature: information_gain(data, feature) for feature in features}
    return max(gains, key=gains.get)

# Function to build the ID3 decision tree
def id3(data):
    # If all target labels are the same, return a leaf node with that label
    if len(data.iloc[:, -1].unique()) == 1:
        return data.iloc[0, -1]

    # If no features left to split on, return the most frequent label
    if len(data.columns) == 1:
        return data.iloc[:, -1].mode()[0]

    # Select the best feature to split on
    feature = best_feature(data)
    tree = {feature: {}}

    # Split the dataset by the feature
    for value in data[feature].unique():
        subset = data[data[feature] == value].drop(columns=[feature])
        tree[feature][value] = id3(subset)

    return tree

# Function to make predictions using the decision tree
def predict(tree, row):
    if not isinstance(tree, dict):  # If it's a leaf node, return the label
        return tree

    feature = next(iter(tree))  # Get the first key (feature name)
    value = row.get(feature)

    if value not in tree[feature]:  # Handle unseen values
        return "Unknown"

    return predict(tree[feature][value], row)

# Example dataset: PlayTennis dataset
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No']
})

# Train the ID3 decision tree on the dataset
tree = id3(data)

# Print the resulting tree
print("Decision Tree:")
print(tree)

# Predict using the trained tree
sample = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Weak'}
prediction = predict(tree, sample)

print(f"Prediction for {sample}: {prediction}")

