# DeepLearning-Logaritmic
DeepLearning-Logaritmic

Bu, Keras kütüphanesi kullanılarak oluşturulmuş bir lojistik regresyon modeli inşa etmek için Python kodudur. Model, Diyabet veri setinde eğitilir ve Pandas veri manipülasyonu, sayısal işlemler için NumPy ve görselleştirme için Matplotlib gibi gerekli kütüphanelerin içe aktarılmasıyla başlar. Veri seti daha sonra Pandas'ın read_csv yöntemi kullanılarak içe aktarılır.

Sonra, bağımsız değişken matrisi ve bağımlı değişken vektörü veri setinden oluşturulur. Veriler, scikit-learn'den train_test_split yöntemi kullanılarak eğitim ve test setlerine ayrılır. Bağımsız değişkenler, scikit-learn'den StandardScaler yöntemi kullanılarak standartlaştırılır.

Keras kütüphanesi daha sonra içe aktarılır ve Sequential sınıfını kullanarak bir Dizin modeli oluşturulur. İki katman add yöntemi kullanılarak modele eklenir. İlk katman, giriş ve gizli katmandır ve ikinci katman çıktı katmanıdır. Giriş ve gizli katmanda 8 nöron bulunur ve doğrultucu aktivasyon fonksiyonu kullanılır. Çıktı katmanı bir nörona sahiptir ve sigmoid aktivasyon fonksiyonunu kullanır.

Son olarak, model, derleme yöntemi kullanılarak derlenir. Derleme yönteminin birkaç parametresi vardır, bunlar arasında optimizör, kayıp fonksiyonu ve metrik fonksiyon yer alır. Optimizör, eğitim sırasında modelin öğrenme hızını kontrol etmek için kullanılır. Kayıp fonksiyonu, tahmin edilen çıktı ile gerçek çıktı arasındaki hatayı hesaplamak için kullanılır. Metrik fonksiyonu, modelin performansını değerlendirmek için kullanılır.








