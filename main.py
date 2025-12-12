
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from helpers import evaluate_and_plot, plot_model_metrics, load_titanic

# --------------------------
# 1. Load dataset
# --------------------------
df = load_titanic()

print("\n=== HEAD ===")
print(df.head())

print("\n=== INFO ===")
print(df.info())

print("\n=== DESCRIBE ===")
print(df.describe())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# --------------------------
# 2. Exploratory Data Analysis (EDA)
# --------------------------
num_features = ['age', 'fare', 'sibsp', 'parch']

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(num_features):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f'{col} Distribution')

plt.tight_layout()
plt.show()

cat_features = ['sex', 'pclass', 'embarked']

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes = axes.flatten()

for i, col in enumerate(cat_features):
    sns.countplot(x=col, data=df, ax=axes[i])
    axes[i].set_title(f'{col} Counts')

plt.tight_layout()
plt.show()


# Correlation
numeric_df = df.select_dtypes(include='number')
corr = numeric_df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr[['survived']], annot=True, cmap='coolwarm')
plt.title("Correlation of Features with Survived")
plt.show()

# --------------------------
# 3. Data Cleaning
# --------------------------
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df = df.drop_duplicates()

df.drop(['deck', 'embark_town', 'alive', 'who', 'adult_male', 'class', 'alone'], axis=1, inplace=True)

# --------------------------
# 4. Feature Engineering
# --------------------------
df['FamilySize'] = df['sibsp'] + df['parch'] + 1
df['IsAlone'] = np.where(df['FamilySize'] == 1, 1, 0)

# --------------------------
# 5. Categorical Encoding
# --------------------------
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['embarked'] = le.fit_transform(df['embarked'])

# --------------------------
# 6. Outlier Handling
# --------------------------
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df['fare'] = df['fare'].clip(lower=lower, upper=upper)

df['fare'] = np.log1p(df['fare'])


# --------------------------
# 7. Scaling / Normalization
# --------------------------
scaler = StandardScaler()
df[['age', 'fare', 'FamilySize']] = scaler.fit_transform(df[['age', 'fare', 'FamilySize']])

# --------------------------
# 8. Train/Test Split
# --------------------------
X = df.drop('survived', axis=1)
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)


# --------------------------
# 9. Modeling
# --------------------------
# Logistic Regression
log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train, y_train)
logreg_pred = log_model.predict(X_test)

# K Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

# --------------------------
# 10. Evaluation
# --------------------------

logreg_metrics = evaluate_and_plot(y_test, logreg_pred, "Logistic Regression")
knn_metrics = evaluate_and_plot(y_test, knn_pred, "K-Nearest Neighbors")

plot_model_metrics(logreg_metrics, knn_metrics)