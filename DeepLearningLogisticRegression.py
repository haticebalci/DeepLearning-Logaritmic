
'''Python içine gerekli kütüphaneler import edilir.Diğer kütüphaneler ilgili işlem yapılmadan önce aşağıda import edilecektir. ''' 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''Kullanacağımız veri seti Diabetes veri setidir.Python'ın pandas kütüphanesinin read_csv methodu ile veri setini import edilir.'''

veriler=pd.read_csv('diabetes.csv')
print(veriler)

'''Veri setinde bağımlı ve bağımsız değişkenlerden oluşan toplamda 9 ayrı kolon bulunmaktadır.Öncelikle veri setinde bulunan değişkenlerden bağımsız 
değişken matrisi ve bağımlı değişken vektörü oluşturulur.'''

x=veriler.iloc[:,:8]#Bağımsız değişkenler
y=veriler.iloc[:,8]#Bağımlı değişken

'''Veri setini eğitmeden önce eğitim ve test kümelerine ayrılır.Train kümelerinde eğitildikten sonra test kümesinde tahminleme yaptırılır.'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

'''Derin öğrenme yönteminde sisteme girdi olarak yüklenecek bağımsız değişkenler öncelikle standardizasyon işlemine tabi tutulur.Bağımsız değişkenlerin 0 ve 1 aralığına gelmesi için 
Standard Scaler methodu ile standardize edilmesi sağlanır.'''

from sklearn.preprocessing import StandardScaler
sts = StandardScaler()
x_train = sts.fit_transform(x_train)
x_test = sts.transform(x_test)

'''Verilerde ön işleme yapılıp derin öğrenme için hazıır hale geldiğinde Yapay sinir ağı oluşturmak için keras kütüphanesini ve aynı şekilde Kullanılacak model ve
katmanları import ederiz.'''

from tensorflow import keras
from keras.models import Sequential #Yapay sinir ağında sıralı katmanlar oluşturmayı sağlayan model olup .add methodu ile katman eklenmesini sağlar.
from keras.layers import Dense #Kerasta katmanları temsil eden sınıftır.


'''İmport edilen keras Sequential modeli üzerinden logr adında bir obje tanımlanır.Bu objenin atanmasıyla yapay sinir ağı oluşturulmuş olur.Ancak burada oluşturulan yapay
 sinir ağının içi boştur.Yapay sinir ağı içerisine .add methodu ile katmanların eklenmesi sağlanır.'''

'''.add methodu ile öncelikle yapay sinir ağına girdi ve gizli katmanı eklenir.Katmanlar için belirli parametreler kullanılır.
-Kernel_initializer=Başlangıç ağırlıklarını belirler.
-İnput_dim=Bağımsız değişken(girdi) sayısı
-units=Gizli katmanda bulunan nöron sayısı
-activation=Katmanda kullanılacak aktivasyon fonksiyonu
Bu veri seti için gizli katmanda rectifier fonksiyonunu kullanırız.'''

logr=Sequential()
logr.add(Dense(units=8, kernel_initializer='uniform',activation='relu',input_dim=8)) #Girdi ve Gizli katmanlar eklenir.
logr.add(Dense(1,activation = 'sigmoid'))#Çıktı katmanında .add(Dense(1, activation='sigmoid')) ifadesindeki 1, çıktı katmanında bulunan nöron sayısını ifade eder. Lojistik regresyon modelinde, çıktı sınıfları sadece iki tane olduğu için, tek bir nöron yeterlidir. Bu nöronun aktivasyon fonksiyonu olarak sigmoid fonksiyonu kullanılmıştır. Sigmoid fonksiyonu, çıktıyı 0 ile 1 arasında bir olasılık değeri olarak verir ve bu nedenle lojistik regresyon sınıflandırma modelinde sıklıkla kullanılır.


'''Modeşi oluşturduktan sonra .compile methodu ile yapay sinir ağının derlenmesi gerekmektedir.Bu method modelin performansını gözlemleyebilmek adına farklı fonksiyonların kullanılmasını sağlar.Compile 
methodu pek çok parametreye sahip olup temelde 3 ana parametresi bulunur.
1)optimizer=Derin öğrenme modellerinde öğrenme oranını kontrol etmektedir.Derin öğrenmede kullanılan Stochastic gradient descent uygulamasını temsil amacıyla adam optimizasyon 
fonksiyonunu kullanıyoruz.Amaç optimum noktanın bulunmasıdır.Model için optimal ağırlıkların ne kadar hızlı hesaplandığını belirler.
2)loss=Loss fonksiyonu ile gerçekte olan değer ile tahmin edilen değer arasındaki farkın yani hatanın hesaplanmasını sağlar.Veri setinde tahmin edilen bağımlı değişken ikiden fazla 
türden oluştuğundan 'binary_crossentropy' fonksiyonunu kullandık.
3)metrics=Modelin başarısını ve performansını görmek için kullanılan parametredir.Örnek veri setinde accuracy methodunu kullandık.'''


logr.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


'''Oluşturulan modelin eğitilmesi için .fit methodu kullanılır.Kullanılan parametreler ;
-batch_size=Eğitim veya test sırasında verilerin yığın yani küme şeklinde eğitime sokulmasını sağlar.
-epoch=Veri setinin modele kaç kez gireceğini belirtmek için kullanılır.'''

logr.fit(x_train,y_train,batch_size=15,epochs=250)

'''Model eğitildikten sonra tahmin ettirme aşamasına geçilir ve predict methodu ile tahminleme yaptırılır.'''

y_pred=logr.predict(x_test)


'''Python Scikit Learn kütüphanesinden model başarısını ölçebileceğemiz accuracy değerini hesaplamamıza yardımcı olacak accuracy_score fonksiyonu import edilir ve 
obje oluşturulur.'''

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_pred.round(), normalize=True)
print(accuracy)
##Başarı oranı:0.77