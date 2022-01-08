from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn import metrics

df = pd.read_csv('https://data.heatonresearch.com/data/t81-558/auto-mpg.csv',
                 na_values=['NA', '?'])

cars = df.name

df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

X = df [['cylinders', 'displacement', 'horsepower', 'weight',
'acceleration', 'year', 'origin']]. values

y = df.mpg.values

model = Sequential()
model.add(Dense(25, input_dim=X.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, y, verbose=2, epochs=100)

y_pred = model.predict(X)

print(np.sqrt(metrics.mean_squared_error(y, y_pred)))

'''
Yukarıdaki kod dört katman içerir. İlk katman girdi katmanıdır,
çünkü programcının veri kümesinin sahip olduğu girdi sayısı olarak ayarladığı input_dim parametresini içerir. 
Ağ, veri setindeki her sütun için bir giriş nöronuna ihtiyaç duyar (kukla değişkenler dahil).

Her biri 25 ve 10 nöron içeren iki katmana sahiptir. 
Tasarlayanların bu sayıları nasıl seçtiği merak edilir. 
Gizli bir nöron yapısının seçilmesi, sinir ağları hakkında en sık sorulan sorulardan biridir. 
Maalesef doğru bir cevap yok. Bunlar hiperparametrelerdir. 
Sinir ağı performansını etkileyebilecek ayarlardır, ancak bunları ayarlamanın açıkça tanımlanmış bir yolu yoktur.

Genel olarak, daha fazla gizli nöron, karmaşık problemlere overfitting yeteneği anlamına gelir. 
Bununla birlikte, çok fazla nöron aşırı uyum ve uzun eğitim sürelerine yol açabilir. 
Çok azı sorunun yetersiz kalmasına neden olabilir ve doğruluktan ödün verir. 
Ayrıca, sahip olduğunuz kaç katman sayısı başka bir hiperparametredir. 
Genel olarak, daha fazla katman, sinir ağının feature engineering ve data preprocessing 
daha fazla gerçekleştirebilmesini sağlar.
Ancak bu aynı zamanda eğitim süreleri ve gereğinden fazla overfitting riski pahasına da geliyor. 
Genel olarak, nöron sayımlarının giriş katmanının yakınında daha büyük başladığını ve bir 
çeşit üçgen şeklinde çıktı katmanına doğru küçülme eğiliminde olduğu görülür.

verbose=0 - İlerleme çıktısı yok 
verbose=1 - İlerleme çubuğunu göster
verbose=2 - Özet ilerleme çıktısı 
'''

