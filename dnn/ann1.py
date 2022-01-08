import pandas as pd
import numpy as np
df = pd.read_csv('dataset.csv')

X = df.iloc[:,0:3].values
y = df.iloc[:,3].values

y = y.reshape(5, 1)



np.random.seed(42)
weights = np.random.rand(3, 1)
bias = np.random.rand(1)
lr = 0.05

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


for epoch in range(20000): # epoch = 20000
    inputs = X
    '''
        İleri beslemenin ilk bölümü
        Burada, girdinin ve ağırlık vektörünün iç çarpımını bulunuyor ve 
        buna bias ekleniyor
    '''
    XW = np.dot(X, weights) + bias
    '''
        İç çarpım, ileri besleme bölümünün 2. adımında açıklandığı gibi 
        sigmoid aktivasyon işlevinden geçirilir. Bu, algoritmanın ileri 
        besleme bölümünü tamamlar. 
    '''
    z = sigmoid(XW)
    '''
        Backpropagation başlıyor. 
        z değişkeni tahmin edilen çıktıları içerir. 
        Backpropagation'ın ilk adımı hatayı bulmaktır. 
        Hata bulunuyor
    '''
    error = z - y
    #hatalar toplamı ekrana yazdırılıyor
    print(error.sum())
    '''
          Backpropagation 2. adımı başlıyor
          d_cost/d_w = d_cost / d_pred * d_pred / dz , d_z/ d_w
          d_cost / d_pred  2 * (predicted - observed) olarak hesaplanabilir.
          Burada 2 sabittir ve bu nedenle göz ardı edilebilir. 
          Bu temelde önceden hesaplanan hatadır.  
    '''
    dcost_dpred = error
    '''
        d_pred / dz bulunuyor.
        Burada "d_pred" basitçe sigmoid fonksiyonudur ve 
        iç çarpım "z" açısından farklılaştırılmıştır.
    '''
    dpred_dz = sigmoid_der(z)
    '''
        d_z/ d_w bulunuyor
        z=x1w1+x2w2+x3w3+b
        Bu nedenle, herhangi bir ağırlığa göre türev basitçe karşılık gelen girdidir. 
        Dolayısıyla, herhangi bir ağırlığa göre maliyet fonksiyonunun nihai türevi şudur:
        egim = input x dcost_dpred x dpred_dz
        Burada, dcost_dpred ve dpred_dz'nin ürününü içeren z_delta değişkeni elde edilir. 
        Her kayıt boyunca döngü yapmak ve girdiyi karşılık gelen z_delta ile çarpmak yerine, 
        girdi özelliği matrisinin devriğini alınıp z_delta ile çarpılır. 
        Son olarak, yakınsama hızını artırmak için öğrenme hızı değişkeni lr türevle çarpılır.
        
    '''
    z_delta = dcost_dpred * dpred_dz
    inputs = X.T
    weights -= lr * np.dot(inputs, z_delta)
    
    for num in z_delta:
        bias -=lr * num
        
        
        
        
        
