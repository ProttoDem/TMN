import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("bioresponse.csv")
X = data.drop("Activity", axis=1)  # Всі ознаки, крім "Activity"
y = data["Activity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
#from sklearn.tree import export_text
#tree_rules = export_text(model, feature_names=X.columns.tolist())
#print(tree_rules)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

# Припустимо, що ви вже маєте модель "model" та отримали передбачення "y_pred" на тестовому наборі.

# 1. Частка правильних відповідей (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 2. Точність (Precision)
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# 3. Повнота (Recall)
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# 4. F1-середнє (F1-score)
f1 = f1_score(y_test, y_pred)
print("F1-score:", f1)

# 5. Log-loss
# Log-loss - це метрика, яка зазвичай використовується для бінарної класифікації. Потрібно мати ймовірності, а не просто бінарні передбачення.
# model.predict_proba повертає ймовірності для кожного класу (0 і 1), отже, потрібно взяти y_pred_proba[:, 1] для класу 1.
y_pred_proba = model.predict_proba(X_test)
log_loss_value = log_loss(y_test, y_pred_proba[:, 1])
print("Log-loss:", log_loss_value)

from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

# Отримайте ймовірності передбачень для класу 1
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Повнота (Recall)')
plt.ylabel('Точність (Precision)')
plt.title('Precision-Recall Крива')
plt.show()

# ROC-крива
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Крива (площа під кривою = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-крива')
plt.legend(loc='lower right')
plt.show()