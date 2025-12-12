import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_titanic(file_path='data/titanic_local.csv'):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print("Loaded Titanic dataset from local CSV.")
    else:
        df = sns.load_dataset('titanic')
        df.to_csv(file_path, index=False)
        print("Downloaded Titanic dataset and saved locally as titanic_local.csv.")
    return df


def evaluate_and_plot(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred, output_dict=True)
    accuracy = accuracy_score(y_true, y_pred)
    recall1 = report['1']['recall']
    f1_1 = report['1']['f1-score']

    print(f"\n--- {model_name} ---")
    print("Accuracy:", round(accuracy, 2))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

    return [accuracy, recall1, f1_1]

def plot_model_metrics(logreg_metrics, knn_metrics, metrics_names=['Accuracy', 'Class 1 Recall', 'Class 1 F1']):
    import numpy as np
    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, logreg_metrics, width, label='Logistic Regression', color='skyblue')
    ax.bar(x + width/2, knn_metrics, width, label='KNN', color='salmon')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.set_title('Model Comparison Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()

    for rects in ax.patches:
        ax.annotate(f'{rects.get_height():.2f}',
                    xy=(rects.get_x() + rects.get_width()/2, rects.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
