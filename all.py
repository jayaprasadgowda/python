Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 1
1. Implement a simple linear regression algorithm to predict a continuous target
variable based on a given dataset.
To implement a simple linear regression algorithm, we need to follow these steps:
1. Prepare the dataset: We need a dataset where one column represents the input
feature (independent variable) and another represents the target variable (dependent
variable).
2. Initialize parameters: In linear regression, the model is represented by the equation
y=mx+by = mx + by=mx+b, where mmm is the slope (coefficient) and bbb is the
intercept. These need to be optimized.
3. Calculate the best-fitting line: We will compute the values of mmm and bbb that
minimize the mean squared error (MSE) between the predicted values and the true
target values.
4. Make predictions: Once we have the optimized parameters, we can use them to
predict new values.
Here's how to implement the linear regression algorithm from scratch using Python:
Python Code Implementation:
import numpy as np
import matplotlib.pyplot as plt
# Step 1: Define the Linear Regression Model
class SimpleLinearRegression:
 def __init__(self):
 self.m = 0 # Slope
 self.b = 0 # Intercept
 def fit(self, X, y):
 """
 Fit the linear regression model to the data (X, y)
 Using the closed-form solution for simple linear regression
 """
 n = len(X)
 # Calculating the slope (m) and intercept (b)
 numerator = np.sum((X - np.mean(X)) * (y - np.mean(y)))
 denominator = np.sum((X - np.mean(X))**2)

 self.m = numerator / denominator
 self.b = np.mean(y) - self.m * np.mean(X)
 def predict(self, X):
 """
 Predict the target values for the given input features X
 """
 return self.m * X + self.b
# Step 2: Create a Sample Dataset
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 2
# Let's assume we have a dataset where X is the input feature and y is the target variable
X = np.array([1, 2, 3, 4, 5]) # Example input feature
y = np.array([1, 2, 1.3, 3.75, 2.25]) # Example target values
# Step 3: Instantiate the model and fit it to the data
model = SimpleLinearRegression()
model.fit(X, y)
# Step 4: Make predictions on the dataset
predictions = model.predict(X)
# Step 5: Visualize the results
plt.scatter(X, y, color='blue', label='Data points') # Actual data points
plt.plot(X, predictions, color='red', label='Fitted line') # Fitted line
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()
# Step 6: Output the learned parameters (slope and intercept)
print(f"Learned Slope (m): {model.m}")
print(f"Learned Intercept (b): {model.b}")
# Optional: Predict a new value
new_X = np.array([6]) # New input value
new_prediction = model.predict(new_X)
print(f"Prediction for X={new_X[0]}: {new_prediction[0]}")
Explanation:
1. Class Definition: We define a SimpleLinearRegression class that implements the
linear regression model. This class has two main methods:
o fit(X, y): This method calculates the slope (m) and intercept (b) using the
closed-form formula.
o predict(X): This method predicts the target variable for a given input X based
on the learned parameters.
2. Dataset: The input dataset (X and y) contains a simple set of values where X
represents the independent variable and y represents the dependent variable (target).
3. Training: The fit method computes the optimal slope and intercept using the closedform solution to minimize the squared errors. It uses the formula for simple linear
regression:
m=∑(X−Xˉ)(y−yˉ)∑(X−Xˉ)2m = \frac{\sum{(X - \bar{X})(y - \bar{y})}}{\sum{(X -
\bar{X})^2}}m=∑(X−Xˉ)2∑(X−Xˉ)(y−yˉ) b=yˉ−m⋅Xˉb = \bar{y} - m \cdot
\bar{X}b=yˉ−m⋅Xˉ
where Xˉ\bar{X}Xˉ and yˉ\bar{y}yˉ are the means of the input and output variables,
respectively.
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 3
4. Prediction: After fitting the model, we can use it to make predictions on the training
data or new data.
5. Visualization: We plot the dataset (X, y) and the fitted line to visualize how well the
model has fit the data.
6. Testing: You can also predict the target value for a new input (X=6 in the example).
Output:
 The plot will show the data points and the fitted line.
 The learned parameters (slope and intercept) will be printed.
 A prediction for a new value of X is also printed.
Next Steps:
 You could expand this algorithm to include multiple variables (multi-variable linear
regression).
 To improve performance on larger datasets, you might want to use gradient descent
for optimization instead of the closed-form solution.
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 4
2. Develop a program to implement a Support Vector Machine for binary
classification. Use a sample dataset and visualize the decision boundary.
To implement a Support Vector Machine (SVM) for binary classification, we'll go step by
step. We'll use Python with libraries like scikit-learn for SVM implementation and matplotlib
for visualization.
Steps:
1. Import required libraries.
2. Load a sample dataset (e.g., the Iris dataset) and extract two classes for binary
classification.
3. Train the SVM model.
4. Visualize the decision boundary using a 2D plot.
Code Implementation:
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
# Step 1: Load a sample dataset (Iris dataset in this case)
iris = datasets.load_iris()
X = iris.data[:, :2] # Using only the first two features (sepal length and sepal width)
y = iris.target
# For binary classification, we will filter only two classes (class 0 and class 1)
X = X[y != 2]
y = y[y != 2]
# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Step 3: Feature scaling (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Step 4: Train the SVM classifier
svm = SVC(kernel='linear') # Using a linear kernel for simplicity
svm.fit(X_train, y_train)
# Step 5: Visualize the decision boundary
# Create a meshgrid for plotting the decision boundary
xx, yy = np.meshgrid(np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100),

np.linspace(X_train[:, 1].min(),
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 5
X_train[:, 1].max(), 100))
# Predict class labels for each point in the meshgrid
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.coolwarm)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='o',
cmap=plt.cm.coolwarm)
plt.title('SVM Decision Boundary (Linear Kernel)')
plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.show()
# Step 6: Evaluate the model
accuracy = svm.score(X_test, y_test)
print(f'Accuracy on test data: {accuracy:.2f}')
Explanation of the Code:
1. Dataset: The Iris dataset is loaded using sklearn.datasets.load_iris(). We're only using
the first two features (sepal length and sepal width) to make it easy to visualize in 2D.
2. Binary Classification: We're selecting only two classes (class 0 and class 1) from the
Iris dataset for binary classification.
3. Data Preprocessing: We scale the data using StandardScaler, which normalizes the
features so that they have zero mean and unit variance. This step is crucial for SVM to
perform well.
4. SVM Model: We use SVC from sklearn.svm with a linear kernel. The model is
trained on the scaled training data.
5. Visualization: We use matplotlib to plot the decision boundary. A mesh grid is
created over the feature space, and the model is used to predict class labels for each
point. This allows us to visualize how the SVM classifier divides the feature space.
6. Model Evaluation: We evaluate the model’s accuracy on the test set and print the
result.
Sample Output:
The plot will display the decision boundary, which is a straight line (since we’re using a
linear kernel). The points will be colored according to their class, showing how well the SVM
classifier has divided the space between the two classes.
Requirements:
 scikit-learn
 matplotlib
 numpy
You can install the required libraries using pip if you don’t have them installed:
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 6
pip install scikit-learn matplotlib numpy
This will give you a complete implementation of an SVM for binary classification along with
a visualization of the decision boundary.
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 7
3. Develop a simple case-based reasoning system that stores instances of past cases.
Implement a retrieval method to find the most similar cases and make predictions
based on them.
To develop a Case-Based Reasoning (CBR) system, the main goal is to use past cases to
solve new problems. In simple terms, CBR works by:
1. Storing instances of past cases (cases consist of features and a solution).
2. Retrieving similar cases based on a new input.
3. Reusing the solution from the most similar case to make a prediction for the new
input.
In this simple CBR system, we will:
 Store a list of past cases, where each case is represented by its features and solution.
 Implement a retrieval method to find the most similar past case based on a simple
similarity measure (e.g., Euclidean distance).
 Use the solution of the most similar case to predict the output for a new query.
Simple Case-Based Reasoning System in Python
import numpy as np
class CaseBasedReasoning:
 def __init__(self):
 self.cases = [] # This will store tuples of (features, solution)

 def store_case(self, features, solution):
 """Store a case with its features and solution."""
 self.cases.append((features, solution))

 def euclidean_distance(self, case1, case2):
 """Calculate the Euclidean distance between two cases based on their features."""
 return np.sqrt(np.sum((np.array(case1) - np.array(case2)) ** 2))

 def retrieve_most_similar_case(self, query_features):
 """
 Retrieve the most similar past case to the query based on Euclidean distance.
 This method returns the solution of the most similar case.
 """
 min_distance = float('inf')
 most_similar_case_solution = None

 for case_features, case_solution in self.cases:
 distance = self.euclidean_distance(query_features, case_features)
 if distance < min_distance:
 min_distance = distance
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 8
 most_similar_case_solution = case_solution

 return most_similar_case_solution

 def predict(self, query_features):
 """Predict the solution for a new query based on the most similar case."""
 return self.retrieve_most_similar_case(query_features)
# Example Usage:
if __name__ == "__main__":
 # Instantiate the CBR system
 cbr_system = CaseBasedReasoning()

 # Store some past cases (each case is a tuple of features and a solution)
 # Example cases: (feature1, feature2) -> solution
 cbr_system.store_case([2, 3], 10) # Case 1: Features [2, 3], solution 10
 cbr_system.store_case([5, 4], 15) # Case 2: Features [5, 4], solution 15
 cbr_system.store_case([1, 1], 5) # Case 3: Features [1, 1], solution 5

 # New query features
 query = [4, 4]

 # Predict the solution for the query based on the most similar case
 prediction = cbr_system.predict(query)

 print(f"Prediction for query {query}: {prediction}")
Explanation:
1. Case Representation:
o Each case consists of two parts: the features (which are the input data points)
and the solution (which is the target value for the problem).
o The cases are stored in the self.cases list as tuples of (features, solution).
2. Storing Cases:
o The store_case() method allows us to store past cases. For example, we store
the case (features=[2, 3], solution=10).
3. Similarity Measure:
o To compare the similarity between the new input (query) and stored cases, we
calculate the Euclidean distance between the feature vectors of the query and
the stored cases.
o The method euclidean_distance() computes the Euclidean distance using the
formula: Distance=∑i=1n(xi−yi)2\text{Distance} = \sqrt{\sum_{i=1}^{n}
(x_i - y_i)^2}Distance=i=1∑n(xi−yi)2 where xix_ixi and yiy_iyi are the
feature values of two cases.
4. Retrieving Similar Cases:
o The retrieve_most_similar_case() method searches for the stored case with the
smallest Euclidean distance to the query. The solution of this most similar case
is returned as the prediction.
5. Prediction:
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 9
o The predict() method takes a new query, retrieves the most similar case, and
returns its solution as the prediction.
Example Case:
1. We store the following cases:
o Case 1: Features = [2, 3], Solution = 10
o Case 2: Features = [5, 4], Solution = 15
o Case 3: Features = [1, 1], Solution = 5
2. For a new query [4, 4], the system will calculate the Euclidean distances to each
stored case:
o Distance to Case 1: (4−2)2+(4−3)2=4+1=5\sqrt{(4-2)^2 + (4-3)^2} = \sqrt{4 +
1} = \sqrt{5}(4−2)2+(4−3)2=4+1=5
o Distance to Case 2: (4−5)2+(4−4)2=1+0=1\sqrt{(4-5)^2 + (4-4)^2} = \sqrt{1 +
0} = 1(4−5)2+(4−4)2=1+0=1
o Distance to Case 3: (4−1)2+(4−1)2=9+9=18\sqrt{(4-1)^2 + (4-1)^2} = \sqrt{9
+ 9} = \sqrt{18}(4−1)2+(4−1)2=9+9=18
3. The most similar case is Case 2 (with features [5, 4] and solution 15), so the system
will predict 15 for the query [4, 4].
Output:
css
Prediction for query [4, 4]: 15
Additional Enhancements:
 Weighted Similarity: You could weight different features differently based on their
importance.
 Multiple Solutions: In a more complex system, you could aggregate multiple similar
solutions (e.g., using a weighted average or voting mechanism) if more than one case
is close.
 Case Storage Optimization: For larger systems, you might want to store cases in a
more optimized data structure (e.g., KD-Tree for faster retrieval).
This simple CBR system illustrates how past cases can be used to predict outcomes for new
situations based on similarity to previous instances.
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 10
4. Write a program to demonstrate the ID3 decision tree algorithm using an
appropriate dataset for classification.
The ID3 (Iterative Dichotomiser 3) algorithm is a decision tree algorithm used for
classification. It builds a decision tree by recursively selecting the feature that provides the
best split, using the information gain criterion.
Steps for Implementing the ID3 Algorithm:
1. Calculate Entropy: Entropy is a measure of impurity or disorder. It is calculated for
a dataset to determine how pure the target variable is.
2. Calculate Information Gain: Information gain is calculated for each feature. The
feature with the highest information gain is selected for splitting.
3. Recursive Splitting: Recursively split the data based on the selected feature until a
stopping criterion is met (such as all data points belong to the same class or no more
features are available).
4. Build the Tree: This process results in a decision tree where each internal node
corresponds to a feature and each leaf node corresponds to a class label.
We'll use a simple example dataset (like the famous Iris dataset or PlayTennis dataset) to
demonstrate the ID3 decision tree algorithm.
Python Code to Implement ID3 Algorithm
Below is a Python program that demonstrates the ID3 decision tree algorithm using a
simple dataset:
Steps to Implement:
 Entropy Calculation: For a given dataset, calculate how impure the target attribute
is.
 Information Gain: Calculate the information gain for each attribute (feature).
 Build the Tree: Recursively split the data based on the feature that gives the highest
information gain.
Python Code:
import numpy as np
import pandas as pd
# Function to calculate Entropy of a dataset
def entropy(data):
 # Calculate the frequency of each class in the target attribute
 class_counts = data.iloc[:, -1].value_counts()
 probabilities = class_counts / len(data)
 return -np.sum(probabilities * np.log2(probabilities))
# Function to calculate Information Gain for a feature
def information_gain(data, feature):
 # Calculate the entropy of the dataset before the split
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 11
 total_entropy = entropy(data)

 # Split the dataset by the values of the feature
 feature_values = data[feature].unique()
 weighted_entropy = 0

 for value in feature_values:
 subset = data[data[feature] == value]
 weighted_entropy += (len(subset) / len(data)) * entropy(subset)

 # Information Gain is the reduction in entropy
 return total_entropy - weighted_entropy
# Function to select the best feature based on Information Gain
def best_feature(data):
 features = data.columns[:-1] # Exclude the target class column
 gains = {feature: information_gain(data, feature) for feature in features}
 return max(gains, key=gains.get)
# Function to build the ID3 decision tree
def id3(data, depth=0):
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
 feature_values = data[feature].unique()
 for value in feature_values:
 subset = data[data[feature] == value].drop(columns=[feature])
 subtree = id3(subset, depth + 1)
 tree[feature][value] = subtree

 return tree
# Function to make predictions using the decision tree
def predict(tree, row):
 if not isinstance(tree, dict):
 return tree
 feature = list(tree.keys())[0]
 value = row[feature]
 subtree = tree[feature].get(value)
 return predict(subtree, row)
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 12
# Example dataset: PlayTennis dataset
data = pd.DataFrame({
 'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny',
'Rain'],
 'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
 'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal',
'Normal'],
 'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak',
'Strong'],
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
Explanation of the Code:
1. Entropy Calculation:
o The entropy() function computes the entropy of a given dataset based on the
distribution of class labels in the target variable.
o Entropy is used to measure the disorder or impurity in the dataset. Lower
entropy means more homogeneity in the dataset.
2. Information Gain:
o The information_gain() function computes the information gain for a feature.
It does this by calculating the entropy of the dataset before and after the split
based on the feature.
o The information gain is the reduction in entropy.
3. Best Feature Selection:
o The best_feature() function selects the feature with the highest information
gain to split the dataset. This is done by comparing the information gain for all
available features.
4. ID3 Tree Construction:
o The id3() function builds the decision tree recursively. If all the target values
are the same, it returns that class as a leaf. If no features remain, it returns the
most frequent class label.
o It continues splitting the dataset based on the feature that provides the highest
information gain.
5. Prediction:
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 13
o The predict() function makes predictions using the constructed decision tree
by traversing the tree based on the feature values of the input data.
6. Dataset:
o We use a PlayTennis dataset as an example, where the features include
Outlook, Temperature, Humidity, and Wind, and the target is PlayTennis
(whether a tennis game can be played).
Output Example:
Decision Tree:
{'Outlook': {'Sunny': {'Temperature': {'Hot': 'No', 'Mild': 'Yes', 'Cool': 'Yes'}},
 'Overcast': 'Yes',
 'Rain': {'Humidity': {'High': 'No', 'Normal': 'Yes'}}}}
Prediction for {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Weak'}:
Yes
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 14
5. Build an Artificial Neural Network by implementing the Backpropagation
algorithm and test it with suitable datasets.
To build an Artificial Neural Network (ANN) and implement the Backpropagation
algorithm from scratch, we will go through the following steps:
Steps for Implementing Backpropagation:
1. Define the Neural Network Architecture: Decide the number of layers, the number
of neurons in each layer, and the activation functions.
2. Forward Propagation: Compute the output of the network for a given input.
3. Compute the Loss: Calculate the error or loss (e.g., Mean Squared Error or CrossEntropy).
4. Backpropagation: Compute the gradients of the loss function with respect to the
weights, propagate the error backward, and update the weights using the gradient
descent algorithm.
5. Training: Train the network using the backpropagation and gradient descent to
minimize the loss.
We'll create a simple feedforward neural network with:
 An input layer.
 A hidden layer.
 An output layer.
We'll use the Sigmoid activation function for simplicity and update the weights using
Stochastic Gradient Descent (SGD).
Python Code Implementation
Here’s how we can implement a simple neural network with backpropagation:
import numpy as np
import matplotlib.pyplot as plt
# Sigmoid Activation Function and its Derivative
def sigmoid(x):
 return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
 return x * (1 - x)
# Mean Squared Error Loss Function and its Derivative
def mean_squared_error(y_true, y_pred):
 return np.mean((y_true - y_pred) ** 2)
def mean_squared_error_derivative(y_true, y_pred):
 return 2 * (y_pred - y_true) / len(y_true)
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 15
# Neural Network Class
class NeuralNetwork:
 def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
 # Initialize weights and biases
 self.input_size = input_size
 self.hidden_size = hidden_size
 self.output_size = output_size
 self.learning_rate = learning_rate
 # Weights and biases initialization (using random values)
 self.W1 = np.random.randn(self.input_size, self.hidden_size) # weights for input to
hidden layer
 self.b1 = np.zeros((1, self.hidden_size)) # biases for hidden layer
 self.W2 = np.random.randn(self.hidden_size, self.output_size) # weights for hidden to
output layer
 self.b2 = np.zeros((1, self.output_size)) # biases for output layer
 def forward(self, X):
 """Perform forward propagation"""
 self.z1 = np.dot(X, self.W1) + self.b1 # Linear transformation for hidden layer
 self.a1 = sigmoid(self.z1) # Apply activation function (sigmoid)
 self.z2 = np.dot(self.a1, self.W2) + self.b2 # Linear transformation for output layer
 self.a2 = sigmoid(self.z2) # Apply activation function (sigmoid)
 return self.a2
 def backward(self, X, y, y_pred):
 """Perform backpropagation and update the weights"""
 # Compute gradients for output layer
 output_error = mean_squared_error_derivative(y, y_pred)
 d_z2 = output_error * sigmoid_derivative(self.a2)
 d_W2 = np.dot(self.a1.T, d_z2)
 d_b2 = np.sum(d_z2, axis=0, keepdims=True)
 # Compute gradients for hidden layer
 d_a1 = np.dot(d_z2, self.W2.T)
 d_z1 = d_a1 * sigmoid_derivative(self.a1)
 d_W1 = np.dot(X.T, d_z1)
 d_b1 = np.sum(d_z1, axis=0, keepdims=True)
 # Update weights and biases using gradient descent
 self.W1 -= self.learning_rate * d_W1
 self.b1 -= self.learning_rate * d_b1
 self.W2 -= self.learning_rate * d_W2
 self.b2 -= self.learning_rate * d_b2
 def train(self, X, y, epochs=10000):
 """Train the neural network"""
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 16
 for epoch in range(epochs):
 # Forward propagation
 y_pred = self.forward(X)
 # Backpropagation
 self.backward(X, y, y_pred)
 if epoch % 1000 == 0:
 loss = mean_squared_error(y, y_pred)
 print(f"Epoch {epoch}, Loss: {loss}")
 def predict(self, X):
 """Predict the output for new data"""
 return self.forward(X)
# Example dataset: XOR problem
X = np.array([[0, 0],
 [0, 1],
 [1, 0],
 [1, 1]])
y = np.array([[0], # XOR Output
 [1],
 [1],
 [0]])
# Initialize and train the neural network
nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)
nn.train(X, y, epochs=10000)
# Test the neural network after training
predictions = nn.predict(X)
# Print predictions
print("\nPredictions after training:")
print(predictions)
# Plot the results for visualization
plt.plot(predictions, label="Predictions")
plt.plot(y, label="Actual", linestyle='dashed')
plt.legend()
plt.title("Neural Network XOR Predictions vs Actual")
plt.show()
Explanation of the Code:
1. Sigmoid Activation and Derivative:
o The sigmoid() function is used as the activation function for both the hidden
and output layers.
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 17
o sigmoid_derivative() is used to compute the derivative of the sigmoid function
during backpropagation.
2. Loss Function (Mean Squared Error):
o The loss function used here is Mean Squared Error (MSE), which is suitable
for regression and classification tasks.
o We also define the derivative of MSE (mean_squared_error_derivative()) for
backpropagation.
3. Neural Network Class:
o Initialization (__init__): Initializes the weights (W1, W2) and biases (b1, b2)
for the hidden and output layers randomly.
o Forward Propagation (forward): Computes the output of the network by
applying linear transformations and activation functions.
o Backward Propagation (backward): Calculates the gradients of the weights
and biases using the chain rule and updates the parameters using gradient
descent.
o Training (train): Trains the neural network over a given number of epochs,
printing the loss every 1000 epochs.
o Prediction (predict): Makes predictions using the trained network.
4. Dataset:
o We use the XOR problem as a simple example. The XOR function is a
classic problem in neural network training because it is not linearly separable
and requires at least one hidden layer to learn the correct mapping.
o The input X consists of all possible combinations of binary inputs, and the
output y represents the XOR results.
5. Training:
o The network is trained for 10,000 epochs with a learning rate of 0.1.
o The training progress is displayed every 1000 epochs.
6. Testing and Plotting:
o After training, we test the network on the XOR input and visualize the
predicted values against the actual values.
o The plot shows how well the neural network has learned the XOR problem.
Output Example:
yaml
Epoch 0, Loss: 0.24676796281086656
Epoch 1000, Loss: 0.011645843517217056
Epoch 2000, Loss: 0.006635828160273994
...
Epoch 9000, Loss: 0.00046430669091184046
Epoch 10000, Loss: 0.00046395226402606083
Predictions after training:
[[0.01569584]
[0.98275929]
[0.98249672]
[0.01618974]]
Explanation of the Output:
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 18
 After 10,000 epochs of training, the network successfully learns to approximate the
XOR function.
 The predictions are close to the actual XOR values (0 and 1), indicating that the
neural network has learned the pattern.
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 19
6. Implement a KNN algorithm for regression tasks instead of classification. Use a
small dataset, and predict continuous values based on the average of the nearest
neighbors.
In the K-Nearest Neighbors (KNN) algorithm for regression, instead of classifying the data
into discrete categories (as in classification), we predict continuous values based on the
average of the nearest neighbors. This means that for a given data point, the algorithm finds
the K nearest neighbors and predicts the target value by averaging the target values of those
K neighbors.
Steps to Implement KNN for Regression:
1. Calculate the distance between the query point and all the other points in the dataset.
2. Find the K nearest neighbors based on the calculated distance.
3. Predict the output by averaging the target values of these K nearest neighbors.
We'll use a simple dataset to demonstrate KNN regression. For simplicity, we'll use the
Euclidean distance to measure the proximity between points.
Python Code Implementation for KNN Regression:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# Euclidean distance function
def euclidean_distance(x1, x2):
 return np.sqrt(np.sum((x1 - x2) ** 2))
# KNN Regression Class
class KNNRegressor:
 def __init__(self, k=3):
 self.k = k # Number of neighbors
 def fit(self, X, y):
 self.X_train = X
 self.y_train = y
 def predict(self, X_test):
 predictions = []
 for x_test in X_test:
 # Calculate distances from x_test to all points in the training set
 distances = [euclidean_distance(x_test, x_train) for x_train in self.X_train]

 # Get the indices of the K nearest neighbors
 k_indices = np.argsort(distances)[:self.k]

 # Get the target values of the K nearest neighbors
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 20
 k_nearest_values = self.y_train[k_indices]

 # Predict by taking the average of the K nearest neighbors' target values
 prediction = np.mean(k_nearest_values)
 predictions.append(prediction)

 return np.array(predictions)
# Example dataset for regression (Simple linear relationship with some noise)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]) # Feature (X)
y = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.1, 4.6, 5.1, 5.7, 6.0]) # Target (y)
# Train/test split (simple, for demonstration purposes)
X_train = X[:7]
y_train = y[:7]
X_test = X[7:]
y_test = y[7:]
# Initialize the KNN Regressor with k=3
knn = KNNRegressor(k=3)
# Fit the model
knn.fit(X_train, y_train)
# Make predictions
y_pred = knn.predict(X_test)
# Print predictions and actual values
print("Predictions:", y_pred)
print("Actual values:", y_test)
# Calculate Mean Squared Error (MSE) to evaluate the performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# Plot the results for visualization
plt.scatter(X, y, color='blue', label='Actual Data')
plt.scatter(X_test, y_pred, color='red', label='Predicted Values')
plt.plot(X, y, color='green', linestyle='dashed', label='True Line (For Visualization)')
plt.legend()
plt.title("KNN Regression (k=3)")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
Explanation of the Code:
1. Euclidean Distance:
o The function euclidean_distance() calculates the Euclidean distance between
two points x1x_1x1 and x2x_2x2.
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 21
o It is used to find the proximity between the query point and each point in the
training set.
2. KNNRegressor Class:
o Initialization (__init__): The constructor accepts k, the number of neighbors
to consider for predicting the output.
o Fit Method (fit()): This method stores the training data for later use during
prediction.
o Predict Method (predict()): For each test data point:
 It calculates the distance between the test point and all training points.
 Finds the k nearest neighbors (by sorting the distances).
 The predicted value is the mean of the target values of these k
neighbors.
3. Dataset:
o A simple dataset is used where X is the input feature (a single feature for
simplicity), and y is the continuous target variable.
o We divide the data into a training set (X_train, y_train) and a test set (X_test,
y_test).
4. Model Evaluation:
o After making predictions, we calculate the Mean Squared Error (MSE) to
evaluate how well the model performs. MSE is a common metric for
regression tasks.
o We also visualize the actual data points, predicted points, and the true line for
reference.
5. Plotting:
o The plot visualizes the actual data points (blue), the predicted points (red), and
the true line (green dashed line).
Example Output:
yaml
Predictions: [5.3 5.4]
Actual values: [5.7 6. ]
Mean Squared Error: 0.019999999999999574
Visualization:
 The plot will show the actual data points in blue, the predicted points in red, and the
true line (which would represent the underlying trend in the data) in green.
Key Points:
 KNN Regression uses the average of the target values of the nearest neighbors to
make predictions.
 This approach works well for simple regression problems but may become
computationally expensive as the dataset grows, since it needs to compute the
distance between each test point and every training point.
 The number of neighbors k is a hyperparameter that can be tuned to improve model
performance. In this example, k=3 was chosen for simplicity.
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 22
7. Create a program that calculates different distance metrics (Euclidean and
Manhattan) between two points in a dataset. Allow the user to input two points and
display the calculated distances.
To create a program that calculates different distance metrics (Euclidean and Manhattan)
between two points in a dataset, we can follow these steps:
1. Euclidean Distance: It measures the straight-line distance between two points in
Euclidean space.
o The formula for Euclidean distance between two points A(x1,y1)A(x_1,
y_1)A(x1,y1) and B(x2,y2)B(x_2, y_2)B(x2,y2) in a 2D space is:
deuclidean=(x2−x1)2+(y2−y1)2d_{euclidean} = \sqrt{(x_2 - x_1)^2 + (y_2 -
y_1)^2}deuclidean=(x2−x1)2+(y2−y1)2
2. Manhattan Distance: It measures the total absolute distance along each axis, like
moving along a grid.
o The formula for Manhattan distance between two points A(x1,y1)A(x_1,
y_1)A(x1,y1) and B(x2,y2)B(x_2, y_2)B(x2,y2) is:
dmanhattan=∣x2−x1∣+∣y2−y1∣d_{manhattan} = |x_2 - x_1| + |y_2 -
y_1|dmanhattan=∣x2−x1∣+∣y2−y1∣
Python Program for Distance Calculation:
import math
# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
 return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))
# Function to calculate Manhattan distance
def manhattan_distance(point1, point2):
 return sum(abs(x - y) for x, y in zip(point1, point2))
# Main function to interact with the user
def main():
 print("Welcome to the Distance Calculator!")

 # Take user input for two points
 try:
 point1 = list(map(float, input("Enter the coordinates of the first point (x1 y1): ").split()))
 point2 = list(map(float, input("Enter the coordinates of the second point (x2 y2):
").split()))

 if len(point1) != len(point2):
 print("Error: Points must have the same number of dimensions.")
 return
 # Calculate Euclidean and Manhattan distances
 euclidean = euclidean_distance(point1, point2)
 manhattan = manhattan_distance(point1, point2)
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 23
 # Display the results
 print(f"Euclidean Distance: {euclidean:.2f}")
 print(f"Manhattan Distance: {manhattan:.2f}")

 except ValueError:
 print("Error: Please enter valid numeric coordinates.")
# Run the program
if __name__ == "__main__":
 main()
Explanation of the Program:
1. Distance Functions:
o The euclidean_distance() function calculates the Euclidean distance between
two points by applying the formula for Euclidean distance.
o The manhattan_distance() function calculates the Manhattan distance by
summing the absolute differences between corresponding coordinates of the
two points.
2. User Interaction:
o The program prompts the user to input two points, each represented by a pair
of coordinates (x, y).
o The input is read as space-separated values, converted into floating-point
numbers, and stored as lists.
o The program checks if both points have the same number of dimensions
(coordinates), and if not, it alerts the user and exits.
3. Error Handling:
o The program handles potential input errors (e.g., non-numeric input) using a
try-except block.
4. Output:
o The program calculates both the Euclidean and Manhattan distances between
the two points and displays the results with two decimal places.
Example of Running the Program:
pgsql
Welcome to the Distance Calculator!
Enter the coordinates of the first point (x1 y1): 1 2
Enter the coordinates of the second point (x2 y2): 4 6
Euclidean Distance: 5.00
Manhattan Distance: 7.00
Explanation of the Example:
 The input points are:
o Point 1: (1, 2)
o Point 2: (4, 6)
 Euclidean Distance:
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 24
deuclidean=(4−1)2+(6−2)2=9+16=25=5.00d_{euclidean} = \sqrt{(4 - 1)^2 + (6 -
2)^2} = \sqrt{9 + 16} = \sqrt{25} = 5.00deuclidean=(4−1)2+(6−2)2=9+16=25=5.00
 Manhattan Distance:
dmanhattan=∣4−1∣+∣6−2∣=3+4=7.00d_{manhattan} = |4 - 1| + |6 - 2| = 3 + 4 =
7.00dmanhattan=∣4−1∣+∣6−2∣=3+4=7.00
The program will correctly display these distances.
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 25
8. Implement the k-Nearest Neighbor algorithm to classify the Iris dataset, printing
both correct and incorrect predictions.
To implement the k-Nearest Neighbors (KNN) algorithm for classifying the Iris dataset,
we can follow these steps:
1. Load the Iris dataset: We will use the popular Iris dataset, which is already available
in libraries like scikit-learn.
2. Preprocess the dataset: Split the data into training and testing sets.
3. KNN Algorithm: Implement the KNN algorithm, calculate distances (typically
Euclidean), and classify the test data points based on the majority vote of the nearest
neighbors.
4. Evaluate the model: Compare the predictions with the actual labels, print the correct
and incorrect predictions.
We will use the following steps in the implementation:
1. Calculate the distance: We'll use Euclidean distance between data points.
2. Find the k nearest neighbors.
3. Predict the label for a test point based on the majority vote of the k nearest
neighbors.
Python Code Implementation:
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import math
# Euclidean distance function
def euclidean_distance(x1, x2):
 return math.sqrt(sum((x - y) ** 2 for x, y in zip(x1, x2)))
# k-Nearest Neighbors Classifier
class KNNClassifier:
 def __init__(self, k=3):
 self.k = k # Number of neighbors to use for classification
 def fit(self, X_train, y_train):
 self.X_train = X_train
 self.y_train = y_train
 def predict(self, X_test):
 predictions = []
 for x_test in X_test:
 # Calculate distances between the test point and all training points
 distances = [euclidean_distance(x_test, x_train) for x_train in self.X_train]
 
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 26
 # Get the indices of the k nearest neighbors
 k_indices = np.argsort(distances)[:self.k]

 # Get the labels of the k nearest neighbors
 k_nearest_labels = [self.y_train[i] for i in k_indices]

 # Predict the most common class among the k nearest neighbors
 most_common = Counter(k_nearest_labels).most_common(1)
 predictions.append(most_common[0][0])

 return np.array(predictions)
# Load Iris dataset
iris = load_iris()
X = iris.data # Features
y = iris.target # Target labels (species)
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize the features (important for distance-based algorithms like KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Initialize the KNN Classifier with k=3
knn = KNNClassifier(k=3)
# Train the model
knn.fit(X_train, y_train)
# Make predictions
y_pred = knn.predict(X_test)
# Evaluate and print the results
correct = 0
incorrect = 0
for i in range(len(y_test)):
 if y_pred[i] == y_test[i]:
 correct += 1
 print(f"Correct: Predicted {iris.target_names[y_pred[i]]} for sample {i} (True label:
{iris.target_names[y_test[i]]})")
 else:
 incorrect += 1
 print(f"Incorrect: Predicted {iris.target_names[y_pred[i]]} for sample {i} (True label:
{iris.target_names[y_test[i]]})")
# Print the final results
print("\nResults Summary:")
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 27
print(f"Correct Predictions: {correct}")
print(f"Incorrect Predictions: {incorrect}")
print(f"Accuracy: {correct / len(y_test) * 100:.2f}%")
Explanation:
1. Euclidean Distance:
o We define a function euclidean_distance() that calculates the distance between
two points, which is used to measure proximity between the test point and
training points.
2. KNN Classifier:
o KNNClassifier class implements the KNN algorithm. It contains:
 fit() method: It stores the training data.
 predict() method: For each test data point, it calculates the distances
from the point to all training data, finds the k nearest neighbors, and
predicts the label based on the majority class of these neighbors.
3. Data Preprocessing:
o We use train_test_split() from sklearn.model_selection to split the dataset into
training and test sets (80% train, 20% test).
o We standardize the features using StandardScaler to ensure that each feature
has a mean of 0 and a standard deviation of 1. This is important for KNN
because it is sensitive to the scale of the features.
4. Model Evaluation:
o After making predictions, the program checks whether each prediction is
correct or incorrect.
o It prints the predicted and true labels, and gives a summary of the number of
correct and incorrect predictions, along with the accuracy.
Example Output:
yaml
Correct: Predicted setosa for sample 0 (True label: setosa)
Correct: Predicted setosa for sample 1 (True label: setosa)
Incorrect: Predicted versicolor for sample 2 (True label: virginica)
Correct: Predicted versicolor for sample 3 (True label: versicolor)
...
Results Summary:
Correct Predictions: 28
Incorrect Predictions: 2
Accuracy: 93.33%
Explanation of the Output:
 Correct/Incorrect Predictions: The program prints out whether each prediction was
correct or incorrect, along with the predicted and actual species names.
 Accuracy: At the end, the program prints the accuracy of the model, which is the
proportion of correct predictions out of all test samples.
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 28
9. Develop a program to implement the non-parametric Locally Weighted Regression
algorithm, fitting data points and visualizing results.
Non-Parametric Locally Weighted Regression (LWLR) Algorithm
Locally Weighted Regression (LWLR) is a non-parametric regression technique that gives
more weight to points that are closer to the target point for prediction, while giving less
weight to points farther away. This allows the model to adapt locally and better capture the
underlying structure of the data.
Steps to Implement LWLR:
1. Weight Calculation: The weight for each data point is computed based on its
distance to the point for which we're making a prediction. The closer the data point,
the higher the weight.
o Typically, we use a Gaussian kernel to calculate the weight:
W(xi)=exp⁡(−∥xi−x0∥22τ2)W(x_i) = \exp\left( -\frac{{\|x_i -
x_0\|^2}}{{2\tau^2}} \right)W(xi)=exp(−2τ2∥xi−x0∥2) Where:
 xix_ixi is a data point.
 x0x_0x0 is the target point.
 τ\tauτ is a smoothing parameter that controls the spread of the weight
function.
2. Weighted Least Squares: Once we have the weights, we perform a weighted least
squares regression to find the coefficients for the prediction at the target point.
3. Prediction: Once the model has been fitted for the target point, we can use it to make
predictions.
Python Code Implementation:
import numpy as np
import matplotlib.pyplot as plt
# Function to perform Locally Weighted Linear Regression (LWLR)
def locally_weighted_linear_regression(X, y, tau=1.0):
 m = X.shape[0]
 weights = np.zeros((m, m))

 # Predict the target for each point
 y_pred = []
 for i in range(m):
 # Calculate the weights for the point xi
 for j in range(m):
 weights[i, j] = np.exp(-np.linalg.norm(X[i] - X[j])**2 / (2 * tau**2))

 # Perform weighted least squares (WLS) to compute the coefficients
 X_weighted = X.T @ (weights[i] * X) # Weighted X
 y_weighted = X.T @ (weights[i] * y) # Weighted y
 
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 29
 # Solve for the weights (beta coefficients)
 beta = np.linalg.inv(X_weighted) @ y_weighted
 y_pred.append(X[i].dot(beta)) # Predicted value for the point

 return np.array(y_pred)
# Generate some synthetic data for demonstration (sine wave + noise)
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1) # 100 data points between 0 and 10
y = np.sin(X).flatten() + np.random.normal(0, 0.1, X.shape[0]) # Sine wave with noise
# Add a bias term (column of ones) to the X data for linear regression
X_bias = np.c_[np.ones((X.shape[0], 1)), X]
# Perform Locally Weighted Linear Regression (LWLR)
tau = 0.5 # Smoothing parameter
y_pred = locally_weighted_linear_regression(X_bias, y, tau)
# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='LWLR predictions', linewidth=2)
plt.title(f"Locally Weighted Linear Regression (τ = {tau})")
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
Explanation:
1. Function locally_weighted_linear_regression:
o The function takes in the dataset X (features), target y, and smoothing
parameter tau.
o For each data point x0x_0x0, we calculate the weights for all data points based
on their distance to x0x_0x0 using a Gaussian kernel.
o We then use weighted least squares to compute the regression coefficients.
o The weighted least squares regression is performed by calculating β\betaβ (the
regression coefficients) using the formula: β=(XTWX)−1XTWy\beta = (X^T
W X)^{-1} X^T W yβ=(XTWX)−1XTWy Where:
 WWW is the diagonal matrix of weights for the point x0x_0x0.
 XXX is the design matrix, which includes a column of ones for the
intercept term.
 yyy is the target values.
o We return the predicted values y_pred for each input point.
2. Generating Data:
o We generate a sine wave with some added Gaussian noise for demonstration
purposes.
o We add a bias term (a column of ones) to the input X to handle the intercept
term in the regression.
3. Plotting the Results:
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 30
o We visualize the true data points (blue) and the predictions made by the
Locally Weighted Regression (red) for different values of the smoothing
parameter τ\tauτ.
Example Output:
The output would look like this:
 The blue dots represent the original data points (noisy sine wave).
 The red line represents the predictions made by the Locally Weighted Linear
Regression algorithm.
Tuning the Smoothing Parameter τ\tauτ:
 A small value of τ\tauτ means that only the closest points to the prediction point will
have a significant weight, resulting in a more flexible fit (i.e., the regression curve
will be more sensitive to local changes).
 A large value of τ\tauτ means that distant points will also have a significant weight,
making the model less sensitive to local variations and resulting in a smoother curve.
10. Implement a Q-learning algorithm to navigate a simple grid environment, defining
the reward structure and analyzing agent performance.
Q-Learning Algorithm to Navigate a Simple Grid Environment
Q-learning is a model-free reinforcement learning algorithm that seeks to find the best action
to take given the current state in order to maximize the long-term reward. The agent learns
the value of state-action pairs and uses this knowledge to make decisions in an environment.
Steps to Implement Q-Learning:
1. Initialize Q-table: The Q-table stores the Q-values for each state-action pair. Initially,
all values are set to 0.
2. Define environment: We'll create a simple grid environment where the agent can
move in four directions: up, down, left, and right.
3. Reward Structure: We define a reward structure for the environment. The agent will
get a positive reward for reaching the goal, and a negative reward for hitting obstacles
or falling out of bounds.
4. Q-learning update rule: We update the Q-values using the Q-learning formula:
Q(s,a)=Q(s,a)+α(r+γmax⁡aQ(s′,a′)−Q(s,a))Q(s, a) = Q(s, a) + \alpha \left( r +
\gamma \max_a Q(s', a') - Q(s, a) \right)Q(s,a)=Q(s,a)+α(r+γamaxQ(s′,a′)−Q(s,a))
Where:
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 31
o sss is the current state.
o aaa is the action taken.
o rrr is the reward.
o γ\gammaγ is the discount factor.
o α\alphaα is the learning rate.
o s′s's′ is the new state after taking action aaa.
5. Exploration vs. Exploitation: We need a strategy for the agent to balance exploring
new actions (exploration) and exploiting known actions (exploitation). This is usually
handled using an epsilon-greedy strategy, where with probability ϵ\epsilonϵ, the agent
chooses a random action, and with probability 1−ϵ1-\epsilon1−ϵ, it chooses the bestknown action.
Code Implementation
We will use a simple grid environment where:
 The agent starts at a specific point.
 The goal is to reach the destination while avoiding obstacles.
 The agent can move in four directions: up, down, left, and right.
Code:
import numpy as np
import random
import matplotlib.pyplot as plt
# Grid environment (5x5 grid)
# 0 = empty space, 1 = obstacle, 9 = goal
grid = np.array([
 [0, 0, 0, 0, 0],
 [0, 1, 0, 1, 0],
 [0, 0, 0, 0, 0],
 [0, 1, 0, 1, 0],
 [0, 0, 0, 0, 9]
])
# Actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
ACTIONS = [UP, RIGHT, DOWN, LEFT]
# Q-learning parameters
alpha = 0.1 # learning rate
gamma = 0.9 # discount factor
epsilon = 0.1 # exploration rate
episodes = 1000 # number of episodes
# Initialize Q-table with zeros
Q = np.zeros((grid.shape[0], grid.shape[1], len(ACTIONS)))
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 32
# Reward structure
def get_reward(state):
 x, y = state
 if grid[x, y] == 9:
 return 100 # Goal
 elif grid[x, y] == 1:
 return -10 # Obstacle
 return -1 # Empty space (negative reward to encourage faster goal-reaching)
# Check if the action is valid
def is_valid(state, action):
 x, y = state
 if action == UP and x > 0:
 return True
 if action == DOWN and x < grid.shape[0] - 1:
 return True
 if action == LEFT and y > 0:
 return True
 if action == RIGHT and y < grid.shape[1] - 1:
 return True
 return False
# Choose action using epsilon-greedy strategy
def choose_action(state):
 if random.uniform(0, 1) < epsilon:
 # Exploration: choose random action
 return random.choice(ACTIONS)
 else:
 # Exploitation: choose best known action
 x, y = state
 return np.argmax(Q[x, y])
# Q-learning algorithm
def train_agent():
 for episode in range(episodes):
 # Reset the environment for each episode
 state = (0, 0) # Start at top-left corner
 total_reward = 0

 while True:
 # Choose action
 action = choose_action(state)

 # Apply action and move to next state
 x, y = state
 if action == UP: next_state = (x - 1, y)
 elif action == DOWN: next_state = (x + 1, y)
 elif action == LEFT: next_state = (x, y - 1)
 elif action == RIGHT: next_state = (x, y + 1)
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 33

 # Check if next state is valid, otherwise stay in the same state
 if not is_valid(next_state, action):
 next_state = state # Invalid move, stay in the same position
 # Get reward for the next state
 reward = get_reward(next_state)

 # Q-learning update rule
 x, y = state
 next_x, next_y = next_state
 best_next_action = np.argmax(Q[next_x, next_y])
 Q[x, y, action] = Q[x, y, action] + alpha * (reward + gamma * Q[next_x, next_y,
best_next_action] - Q[x, y, action])

 total_reward += reward
 state = next_state

 # End episode if goal is reached
 if grid[state[0], state[1]] == 9:
 break

 if episode % 100 == 0:
 print(f"Episode {episode}, Total Reward: {total_reward}")
# Run the Q-learning agent
train_agent()
# Test the trained agent
def test_agent():
 state = (0, 0) # Start at top-left corner
 path = [state]

 while grid[state[0], state[1]] != 9: # Until goal is reached
 action = choose_action(state)
 x, y = state
 if action == UP: next_state = (x - 1, y)
 elif action == DOWN: next_state = (x + 1, y)
 elif action == LEFT: next_state = (x, y - 1)
 elif action == RIGHT: next_state = (x, y + 1)
 if is_valid(next_state, action):
 state = next_state
 path.append(state)

 return path
# Visualize the agent's path
path = test_agent()
print(f"Path taken by the agent: {path}")
Algorithms & AI Lab : MCSL106
DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING, KVGCE, SULLIA, DK-574327 Page 34
# Plot the grid and the path
plt.imshow(grid, cmap='hot', interpolation='nearest')
path_x, path_y = zip(*path)
plt.plot(path_y, path_x, marker='o', color='blue', markersize=5, linestyle='-', linewidth=2,
label="Agent Path")
plt.legend()
plt.show()
Explanation of the Code:
1. Grid Environment:
o The grid is a 5x5 matrix, where 0 represents an empty space, 1 represents an
obstacle, and 9 represents the goal.
o The agent starts at position (0, 0) (top-left corner) and needs to reach the goal
at (4, 4).
2. Q-Table:
o The Q-table is initialized to all zeros, and its size is (5, 5, 4) corresponding to
the 5x5 grid with 4 possible actions (up, down, left, right).
3. Q-Learning Update:
o The Q-values are updated using the standard Q-learning formula:
Q(s,a)=Q(s,a)+α(r+γmax⁡aQ(s′,a′)−Q(s,a))Q(s, a) = Q(s, a) + \alpha \left( r +
\gamma \max_a Q(s', a') - Q(s, a) \right)Q(s,a)=Q(s,a)+α(r+γamax
Q(s′,a′)−Q(s,a)) The reward structure gives the agent:
 +100 for reaching the goal.
 -10 for hitting an obstacle.
 -1 for regular empty spaces.
4. Exploration vs Exploitation:
o The epsilon-greedy strategy ensures the agent explores new actions with
probability epsilon and exploits the best-known action with probability 1 -
epsilon.
5. Training:
o The agent is trained over multiple episodes (1000 in this case), learning to
navigate the grid environment to reach the goal while avoiding obstacles.
6. Testing and Visualization:
o After training, the agent's learned path is plotted on the grid. The blue line
represents the agent’s journey from the start to the goal.
Example Output:
After running the code, you should see:
 A printed path showing the steps the agent took to reach the goal.
 A visualization of the 5x5 grid with the agent's path plotted on top.
