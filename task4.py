import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Завантаження та підготовка даних
# Зауважте, що вам потрібно буде замінити цей крок своїми реальними даними
data = pd.read_csv('electronic_business_security_data.csv')  # Замініть 'your_data.csv' на ваш файл з даними

# Стандартизація даних
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Застосування PCA
pca = PCA(n_components=2)  # Зменшення до 2 компонентів для візуалізації
principal_components = pca.fit_transform(scaled_data)

# Конвертація результатів PCA у DataFrame для зручності
pca_df = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])

# Візуалізація результатів
plt.figure(figsize=(8,6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
plt.xlabel('Перший головний компонент (PC1)')
plt.ylabel('Другий головний компонент (PC2)')
plt.title('PCA результати')
plt.show()
