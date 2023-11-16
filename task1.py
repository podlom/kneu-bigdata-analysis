import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys


def main():
    # Перевірка, чи було передано аргументи командного рядка
    if len(sys.argv) < 2:
        # Отримання імені поточного скрипта
        script_file_name = os.path.basename(sys.argv[0])
    
        print("Використання: python3 ", script_file_name, " filename.csv")
        sys.exit(1)  # Вихід із скрипта з помилкою

    # Отримання назви файлу з аргументів командного рядка
    filename = sys.argv[1]

    # Тут ваш код для обробки CSV файлу
    try:
        # Припустимо, що 'df' - це ваш DataFrame з даними
        df = pd.read_csv(filename)

        # Розділення даних на навчальні та тестові набори
        X = df[['investment_in_security', 'security_level', 'cyber_attack_intensity']]
        y = df['business_profitability']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Створення моделі лінійної регресії
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Оцінка моделі
        print("Коефіцієнти моделі:", model.coef_)
        print("Точність моделі:", model.score(X_test, y_test))

	# Візуалізація результатів
        plt.scatter(model.predict(X_test), y_test)
        plt.xlabel('Прогнозовані значення')
        plt.ylabel('Фактичні значення')
        plt.title('Лінійна регресія для фінансової ефективності')
        plt.show()
    except Exception as e:
        print(f"Помилка при читанні файлу {filename}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

