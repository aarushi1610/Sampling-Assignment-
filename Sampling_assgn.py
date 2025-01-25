import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids

# Step 1: Load the dataset
url = "https://github.com/AnjulaMehto/Sampling_Assignment/raw/main/Creditcard_data.csv"
dataset = pd.read_csv(url)

# Step 2: Balance the dataset using RandomUnderSampler
features = dataset.drop('Class', axis=1)
labels = dataset['Class']
balancer = RandomUnderSampler(sampling_strategy='majority')
balanced_features, balanced_labels = balancer.fit_resample(features, labels)

# Step 3: Apply five sampling techniques on five ML models

# Sampling 1: Random Over-sampling with RandomForest
model1 = RandomForestClassifier()
oversampler = RandomOverSampler()
X_sampled1, y_sampled1 = oversampler.fit_resample(balanced_features, balanced_labels)
model1.fit(X_sampled1, y_sampled1)
accuracy1 = accuracy_score(y_sampled1, model1.predict(X_sampled1))
print(f"RandomForest with Random Over-sampling: {accuracy1*100:.2f}%")

# Sampling 2: Stratified Sampling with GradientBoosting
splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index2, test_index2 = splitter2.split(balanced_features, balanced_labels).__next__()
X_train2, X_test2 = balanced_features.iloc[train_index2], balanced_features.iloc[test_index2]
y_train2, y_test2 = balanced_labels.iloc[train_index2], balanced_labels.iloc[test_index2]

model2 = GradientBoostingClassifier()
model2.fit(X_train2, y_train2)
accuracy2 = accuracy_score(y_test2, model2.predict(X_test2))
print(f"GradientBoosting with Stratified Sampling: {accuracy2*100:.2f}%")

# Sampling 3: Cluster Sampling with LogisticRegression
model3 = LogisticRegression(max_iter=1000)
cluster_sampler = ClusterCentroids(sampling_strategy='majority')
X_sampled3, y_sampled3 = cluster_sampler.fit_resample(balanced_features, balanced_labels)
model3.fit(X_sampled3, y_sampled3)
accuracy3 = accuracy_score(y_sampled3, model3.predict(X_sampled3))
print(f"LogisticRegression with Cluster Sampling: {accuracy3*100:.2f}%")

# Sampling 4: Bootstrap Sampling with SVC
model4 = SVC()
num_splits = 5  # Number of splits for StratifiedKFold
sampling4 = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
total_accuracy4 = 0
for _ in range(2):  # Perform sampling twice
    for train_index4, test_index4 in sampling4.split(balanced_features, balanced_labels):
        X_train4, X_test4 = balanced_features.iloc[train_index4], balanced_features.iloc[test_index4]
        y_train4, y_test4 = balanced_labels.iloc[train_index4], balanced_labels.iloc[test_index4]
        model4.fit(X_train4, y_train4)
        predictions4 = model4.predict(X_test4)
        total_accuracy4 += accuracy_score(y_test4, predictions4)
accuracy4 = total_accuracy4 / (num_splits * 2)
print(f"SVC with Bootstrap Sampling: {accuracy4*100:.2f}%")

# Sampling 5: Stratified K-fold Cross-validation with KNeighborsClassifier
model5 = KNeighborsClassifier()
splitter5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_index5, test_index5 = splitter5.split(balanced_features, balanced_labels).__next__()
X_train5, X_test5 = balanced_features.iloc[train_index5], balanced_features.iloc[test_index5]
y_train5, y_test5 = balanced_labels.iloc[train_index5], balanced_labels.iloc[test_index5]

model5.fit(X_train5, y_train5)
accuracy5 = accuracy_score(y_test5, model5.predict(X_test5))
print(f"KNeighborsClassifier with Stratified K-fold Cross-validation: {accuracy5*100:.2f}%")
