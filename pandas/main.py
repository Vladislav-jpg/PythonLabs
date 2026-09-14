import pandas as pd
series_example = pd.Series([1, 2, 3, 4, 5])
print(series_example)

city = {'Город': ['Москва', 'Санкт-Петербург',
'Новосибирск', 'Екатеринбург'],
 'Год основания': [1147, 1703, 1893, 1723],
 'Население': [11.9, 4.9, 1.5, 1.4]}
df = pd.DataFrame(city)
print(df)

