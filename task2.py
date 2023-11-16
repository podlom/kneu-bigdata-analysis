import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# Завантаження даних
data = pd.read_csv('dataset.csv')  # Припустимо, що у нас є такий файл

# Перетворення даних
X = data.drop('Result', axis=1)  # Всі характеристики
y = data['Result']  # Мітки класів (фішинговий/легітимний)

# Розділення даних на тренувальні та тестові набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Створення моделі класифікатора
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Перевірка ефективності моделі
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))
