import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib  # Use joblib for model serialization
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Feature normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply t-SNE to the training data
tsne = TSNE(n_components=2, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_scaled)

# Visualize the t-SNE result with gender labels
plt.figure(figsize=(10, 6))
colors = {'m': 'blue', 'f': 'red'}
for gender in np.unique(y_train):
    indices = y_train == gender
    plt.scatter(X_train_tsne[indices, 0], X_train_tsne[indices, 1], c=colors[gender], label=gender, alpha=0.6, edgecolors='w', s=60)

plt.title("t-SNE visualization of the training data")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True)
plt.show()

# Function to calculate BIC for KMeans
def calculate_bic(kmeans_model, X):
    # Compute log likelihood (log of the likelihood L)
    log_likelihood = -kmeans_model.score(X)
    
    # Number of clusters
    k = kmeans_model.n_clusters
    
    # Number of parameters p = k (number of clusters) + k*2 (centroids) + k (cluster sizes)
    p = k + k*2 + k
    
    # Number of samples
    n = X.shape[0]
    
    # Calculate BIC
    bic = -2 * log_likelihood + p * np.log(n)
    
    return bic

# Range of k values to evaluate
k_values = range(2, 16)  # Adjust range as needed

# Initialize lists to store BIC values
bic_values = []

# Iterate over each k value
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_scaled)
    
    # Calculate BIC for the current KMeans model
    bic = calculate_bic(kmeans, X_train_scaled)
    
    # Store BIC value
    bic_values.append(bic)

# Plotting BIC values
plt.figure(figsize=(10, 6))
plt.plot(k_values, bic_values, marker='o', linestyle='-', color='b', markersize=8)
plt.title('Bayesian Information Criterion (BIC) for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('BIC Value')
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()

# Finding the optimal k based on BIC
optimal_k_index = np.argmin(bic_values)
optimal_k = k_values[optimal_k_index]
plt.annotate('Optimal k', xy=(optimal_k, bic_values[optimal_k_index]),
             xytext=(optimal_k + 1, bic_values[optimal_k_index] + 100),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=12, color='black')

plt.show()

print(f"Optimal number of clusters (k) based on BIC: {optimal_k}")

# Now use the optimal_k to perform KMeans clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels_train = kmeans.fit_predict(X_train_scaled)
cluster_labels_test = kmeans.predict(X_test_scaled)

# Apply t-SNE to the training data again (if not already done)
X_train_tsne = tsne.fit_transform(X_train_scaled)

# Visualize the t-SNE result with cluster labels
plt.figure(figsize=(10, 6))
for cluster in np.unique(cluster_labels_train):
    indices = cluster_labels_train == cluster
    plt.scatter(X_train_tsne[indices, 0], X_train_tsne[indices, 1], label=f'Cluster {cluster}', alpha=0.6, edgecolors='w', s=60)

plt.title("t-SNE visualization of the training data with clusters")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True)
plt.show()

# One-hot encoding of cluster labels
encoder = OneHotEncoder(categories='auto', sparse=False)
cluster_features_train = encoder.fit_transform(cluster_labels_train.reshape(-1, 1))
cluster_features_test = encoder.transform(cluster_labels_test.reshape(-1, 1))

# Augment original features with cluster features
X_augmented_train = np.hstack((X_train_scaled, cluster_features_train))
X_augmented_test = np.hstack((X_test_scaled, cluster_features_test))

# Define SVM classifier
svm = SVC(kernel='rbf', class_weight='balanced', gamma='scale')

# Training SVM model
svm.fit(X_augmented_train, y_train)

# Save the trained model
joblib.dump(svm, 'svm_model.pkl')

# Evaluation
y_pred = svm.predict(X_augmented_test)

# Overall accuracy
acc = accuracy_score(y_test, y_pred)
print("Overall Accuracy: {:.4f}".format(acc))

# Accuracy per class
acc_m = accuracy_score(y_test[y_test=='m'], y_pred[y_test=='m'])
acc_f = accuracy_score(y_test[y_test=='f'], y_pred[y_test=='f'])
print("Male Accuracy: {:.4f}".format(acc_m))
print("Female Accuracy: {:.4f}".format(acc_f))

# Load the saved model
loaded_model = joblib.load('svm_model.pkl')

# Example: Predict using the loaded model
y_pred_loaded = loaded_model.predict(X_augmented_test)

# Example: Evaluate the loaded model
acc_loaded = accuracy_score(y_test, y_pred_loaded)
print("Overall Accuracy (Loaded Model): {:.4f}".format(acc_loaded))
