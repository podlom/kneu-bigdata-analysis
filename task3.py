import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime


# Функція для перетворення дати в число
def convert_date_to_ordinal(date):
    return datetime.strptime(date, '%Y-%m-%d').toordinal()

# Завантаження даних
data = pd.read_csv('security_incidents.csv')

# Перетворення дати в числовий формат
data['detection_time'] = data['detection_time'].apply(convert_date_to_ordinal)

# Підготовка даних
features = data[['severity', 'impact', 'detection_time']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Кластеризація
kmeans = KMeans(n_clusters=3)
kmeans.fit(scaled_features)
clusters = kmeans.predict(scaled_features)

# Додавання результатів кластеризації до датасету
data['cluster'] = clusters

# Візуалізація (може вимагати додаткової налаштування залежно від даних)
plt.scatter(data['severity'], data['impact'], c=data['cluster'], cmap='viridis')
plt.xlabel('Severity')
plt.ylabel('Impact')
plt.show()

