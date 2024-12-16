# Kütüphaneler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Veri yükleme
df = pd.read_csv('antalya_kiralik_ev.csv')
df.drop("Unnamed: 0", axis=1, inplace=True)

# Aynı olan satırları silme
df.drop_duplicates(inplace=True)

# Fiyata aidat ekleme
df['fiyat'] = df['fiyat'] + df['aidat']

# Aykırı değerleri veri setinden çıkarma
plt.figure()
sns.boxplot(x=df['fiyat'])
plt.title("Fiyat Dağılımı ve Aykırı Değerler")
plt.show()
outliers_fiyat = df[['fiyat']].quantile(q=.991) # Üst sınır belirliyoruz.
print(outliers_fiyat)
# üst sınırdan (96939) yüksek olanları veri setinden çıkartıyoruz.
df2 = df[df['fiyat']<outliers_fiyat[0]]

# Bina yaşını numerik değerlere çevirme
df2['bina_yas'] = df2['bina_yas'].replace({
    '0': 0,
    '1-5 arası': 3,
    '5-10 arası': 7.5,
    '11-15 arası': 13,
    '16-20 arası': 18,
    '21-25 arası': 23,
    '26-30 arası': 28,
    '31 ve üzeri': 35
}).astype(int)

# Oda sayısı sütununu, Oda ve Salon sayısı sütunlarına bölme
df2['oda_sayisi'] = df2['oda_sayisi'].replace('Stüdyo (1+0)', '1+0')
df2[['oda_sayisi2', 'salon_sayisi']] = df2['oda_sayisi'].str.split('+', expand=True).astype(float)

# Net alan ve brüt alanı kaldırıp yerine net/brüt oranı ekleme
df2['net_brut_orani'] = df2['net_alan_m2'] / df2['brut_alan_m2'] #Kullanılabilirlik

# Düzenlenen kolonların eski hallerini silme
df2.drop(['aidat','net_alan_m2','brut_alan_m2','oda_sayisi'], axis=1, inplace=True)

plt.title("Daire Sayısı - Binanın Kat Sayısı Grafiği")
plt.xlabel("Bina Kat Sayısı")
plt.ylabel("Daire Sayısı")
plt.grid(True)
n, bins, patches = plt.hist(df2['bina_kat_sayisi'], bins=30, facecolor='blue', alpha=0.75)

# 'Dairenin bulunduğu kat' değerlerini düzenleme
df2['dairenin_bulundugu_kat'] = df2['dairenin_bulundugu_kat'].replace({
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '10': 10,
    '11': 11,
    '12': 12,
    '13': 13,
    '15': 15,
    'Yüksek Giriş': 1,
    'Giriş Katı': 1,
    'Bahçe Katı': 1,
    'Zemin Kat': 1,
    'Bodrum Kat': -1,
    'Giriş Altı Kot 2': -1,
    'Villa Tipi': 2,
    'Çatı Katı': 4
}).astype(int)

# 'Isıtma' değerlerini en iyiden en kötüye sıralayarak sayısal değere dönüştürme
df2['isitma_turu'] = df2['isitma_turu'].replace({
    'Yerden Isıtma': 5,
    'Güneş Enerjisi': 5,
    'Şömine': 5,
    'Klima': 4,
    'Kombi (Doğalgaz)': 3,
    'Merkezi (Pay Ölçer)': 3,
    'Kombi (Elektrik)': 3,
    'Merkezi': 2,
    'Kat Kaloriferi': 2,
    'Fancoil Ünitesi':2,
    'Doğalgaz Sobası': 2,
    'Soba': 1,
    'Yok': 0
}).astype(int)

# 'Otopark' değerlerini en iyiden en kötüye sıralayarak sayısal değere dönüştürme
df2['otopark'] = df2['otopark'].replace({
    'Açık Otopark': 1,
    'Açık & Kapalı Otopark': 3, 
    'Kapalı Otopark': 2,
    'Yok': 0
}).astype(int)

# Sayısallaştıramadığımız kategorik verileri(mahalle) etiketleme işlemi (Label Encoding)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df2['mahalle'] = le.fit_transform(df2.mahalle)

# Sütunları yeniden sıralama
new_column_order = [
    'mahalle', 'isitma_turu', 'otopark', 'esya_durumu', 'sahibi', 'balkon', 'asansor', 'site_icinde',
    'dairenin_bulundugu_kat', 'oda_sayisi2', 'salon_sayisi', 'depozito', 'banyo_sayisi', 
    'bina_kat_sayisi', 'bina_yas', 'net_brut_orani','fiyat']
df2 = df2[new_column_order]

# Çoklu Doğrusal Regresyonun şartı olan Doğrusallığı kontrol etme (Depozito)
plt.figure(figsize=(12, 8))
sns.scatterplot(x=df['depozito'], y=df['fiyat'])
plt.title('Fiyat-Depozito Saçılım Grafiği')
plt.show()

# Makine Öğrenmesi Modellerinin Uygulanması
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Bağımlı ve bağımsız değişkenleri belirleme
y = df2['fiyat']
X = df2.drop('fiyat', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=144)
# Linear Regression
model_LR = LinearRegression()
model_LR.fit(X_train, y_train)
# Ridge Regression
model_Ridge = Ridge()
model_Ridge.fit(X_train, y_train)
# Lasso Regression
model_Lasso = Lasso()
model_Lasso.fit(X_train, y_train)
# XGB Regression
xgb1 = XGBRegressor(colsample_bytree = 0.5, learning_rate = 0.09, max_depth = 4, n_estimators = 2000)
model_xgb = xgb1.fit(X_train, y_train)

# Test seti sonuçlarının tahmin edilmesi
y_pred_LR = model_LR.predict(X_test)
y_pred_Ridge = model_Ridge.predict(X_test)
y_pred_Lasso = model_Lasso.predict(X_test)
y_pred_XGB = model_xgb.predict(X_test)

# Sonuçların Görüntülenmesi
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
print('*Linear Regression*')
print("R-Kare: ",r2_score(y_test, y_pred_LR))
print("MAE: ",mean_absolute_error(y_test, y_pred_LR))
print("MSE: ",mean_squared_error(y_test, y_pred_LR))
print("MedAE: ",median_absolute_error(y_test, y_pred_LR))

print('*Ridge Regression*')
print("R-Kare: ",r2_score(y_test, y_pred_Ridge))
print("MAE: ",mean_absolute_error(y_test, y_pred_Ridge))
print("MSE: ",mean_squared_error(y_test, y_pred_Ridge))
print("MedAE: ",median_absolute_error(y_test, y_pred_Ridge))

print('*Lasso Regression*')
print("R-Kare: ",r2_score(y_test, y_pred_Lasso))
print("MAE: ",mean_absolute_error(y_test, y_pred_Lasso))
print("MSE: ",mean_squared_error(y_test, y_pred_Lasso))
print("MedAE: ",median_absolute_error(y_test, y_pred_Lasso))

print('*XGBoost Regression*')
print("R-Kare: ",r2_score(y_test, y_pred_XGB))
print("MAE: ",mean_absolute_error(y_test, y_pred_XGB))
print("MSE: ",mean_squared_error(y_test, y_pred_XGB))
print("MedAE: ",median_absolute_error(y_test, y_pred_XGB))

# Özniteliklerin Önem Katsayıları
importance = pd.DataFrame({"Importance": model_xgb.feature_importances_},index=X_train.columns)
print(importance)

# Arayüz(Tkinter)
import tkinter as tk
from tkinter.ttk import Combobox
root = tk.Tk()
root.geometry("1700x900")  # Pencere boyutu
root.title("Antalya-Muratpaşa Ev Kirası Fiyat Tahmini")
root.configure(bg="#f0f8ff")
root.state("normal")

# Butonlara Tıklandığında Çalışan Fonksiyonlar
def olumsuz():
    tk.messagebox.showwarning(title="Dikkat!", message="Hatalı veya eksik seçim yaptınız!")

def mahalle_duzenle():
    global mahalle 
    mahalle_deger = mahalle_kutu.get()
    mahalle_harita = {
        "Altındağ": 0, "Bahçelievler": 1, "Balbey": 2, "Bayındır": 3, "Cumhuriyet": 4, 
        "Demircikara": 5, "Deniz": 6, "Doğuyaka": 7, "Dutlubahçe": 8, "Elmalı": 9, 
        "Ermenek": 10, "Etiler": 11, "Fener": 12, "Gebizli": 13, "Gençlik": 14, 
        "Güvenlik": 15, "Güzelbağ": 16, "Güzeloba": 17, "Güzeloluk": 18, "Haşimişcan": 19, 
        "Konuksever": 20, "Kırcami": 21, "Kızılarık": 22, "Kızılsaray": 23, "Kızıltoprak": 24, 
        "Kışla": 25, "Mehmetçik": 26, "Meltem": 27, "Memurevleri": 28, "Meydankavağı": 29, 
        "Muratpaşa": 30, "Yeşilbahçe": 31, "Selçuk": 32, "Sinan": 33, "Soğuksu": 34, 
        "Tahılpazarı": 35, "Topçular": 36, "Varlık": 37, "Yenigöl": 38, "Yenigün": 39, 
        "Sedir": 40, "Yeşildere": 41, "Yeşilova": 42, "Yüksekalan": 43, "Yıldız": 44, 
        "Zerdalilik": 45, "Çaybaşı": 46, "Çağlayan": 47, "Üçgen": 48, "Şirinyalı": 49}
    if mahalle_deger in mahalle_harita:
        mahalle = mahalle_harita[mahalle_deger]
    else:
        olumsuz()
    print(mahalle)

def isitma_duzenle():
    global isitma 
    isitma_deger = isitma_kutu.get()
    isitma_harita = {
         'Yerden Isıtma': 5, 'Güneş Enerjisi': 5, 'Şömine': 5, 'Klima': 4,
         'Kombi (Doğalgaz)': 3, 'Merkezi (Pay Ölçer)': 3, 'Kombi (Elektrik)': 3,
         'Merkezi': 2, 'Kat Kaloriferi': 2, 'Fancoil Ünitesi':2,
         'Doğalgaz Sobası': 2, 'Soba': 1, 'Yok': 0}
    if isitma_deger in isitma_harita:
        isitma = isitma_harita[isitma_deger]
    else:
        olumsuz()
    print(isitma)

def otopark_duzenle():
    global otopark 
    otopark_deger = otopark_kutu.get()
    otopark_harita = {
         'Açık & Kapalı Otopark': 3, 'Kapalı Otopark': 2, 'Açık Otopark': 1, 'Yok': 0}
    if otopark_deger in otopark_harita:
        otopark = otopark_harita[otopark_deger]
    else:
        olumsuz()
    print(otopark)

def esya_durumu_duzenle():
    global esya
    esya_deger = esya_kutu.get()
    if(esya_deger == "Eşyasız"):
        esya = 0
    elif(esya_deger == "Eşyalı"):
        esya = 1
    else:
        olumsuz()
    print(esya)

def sahibi_duzenle():
    global sahibi
    sahibi_deger = sahibi_kutu.get()
    if(sahibi_deger == "Bireysel"):
        sahibi = 0
    elif(sahibi_deger == "Emlakçı"):
        sahibi = 1
    else:
        olumsuz()
    print(sahibi)

def balkon_duzenle():
    global balkon
    balkon_deger = balkon_kutu.get()
    if(balkon_deger == "Yok"):
        balkon = 0
    elif(balkon_deger == "Var"):
        balkon = 1
    else:
        olumsuz()
    print(balkon)

def asansor_duzenle():
    global asansor
    asansor_deger = asansor_kutu.get()
    if(asansor_deger == "Yok"):
        asansor = 0
    elif(asansor_deger == "Var"):
        asansor = 1
    else:
        olumsuz()
    print(asansor)

def site_duzenle():
    global site
    site_deger = site_kutu.get()
    if(site_deger == "Site Dışında"):
        site = 0
    elif(site_deger == "Site İçinde"):
        site = 1
    else:
        olumsuz()
    print(site)

def dairenin_kati_duzenle():
    global kat
    kat_ent = int(kat_entry.get())
    if(17 > kat_ent > 0):
        kat = kat_ent
        print(kat)
    else:
        olumsuz()

def oda_duzenle():
    global oda
    oda_ent = int(oda_entry.get())
    if(9 > oda_ent > 0):
        oda = oda_ent
        print(oda)
    else:
        olumsuz()

def salon_duzenle():
    global salon
    salon_ent = int(salon_entry.get())
    if(4 > salon_ent > 0):
        salon = salon_ent
        print(salon)
    else:
        olumsuz()

def depozito_duzenle():
    global depozito
    depozito_ent = int(depozito_entry.get())
    if(250001 > depozito_ent > 0):
        depozito = depozito_ent
        print(depozito)
    else:
        olumsuz()

def banyo_duzenle():
    global banyo
    banyo_ent = int(banyo_entry.get())
    if(5 > banyo_ent > 0):
        banyo = banyo_ent
        print(banyo)
    else:
        olumsuz()

def bina_kat_duzenle():
    global bina_kat
    bina_kat_ent = int(salon_entry.get())
    if(19 > bina_kat_ent > 0):
        bina_kat = bina_kat_ent
        print(bina_kat)
    else:
        olumsuz()

def yas_duzenle():
    global yas
    yas_ent = int(yas_entry.get())
    if(46 > yas_ent > 0):
        yas = yas_ent
        print(yas)
    else:
        olumsuz()

def alan_duzenle():
    global alan
    alan_ent = float(alan_entry.get())
    if(1 >= alan_ent > 0):
        alan = alan_ent
        print(alan)
    else:
        olumsuz()

# Tkinter Tasarım Kısmı
# Başlık
title_label = tk.Label(root, text="Antalya-Muratpaşa Ev Kirası Fiyat Tahmini", 
                       bg="#e6f2ff",borderwidth=20, padx = 250, pady = 40,
                       font=("Helvetica", 32, "bold"))
title_label.place(x = 70 ,y = 20)

# Mahalle
mahalle_label = tk.Label(text = "Mahalle:", bg="#f0f8ff", font=("Helvetica", 12))
mahalle_label.place(x=100, y=200)
mahalleler = ['Altındağ', 'Bahçelievler', 'Balbey', 'Bayındır', 'Cumhuriyet',
              'Demircikara', 'Deniz', 'Doğuyaka', 'Dutlubahçe', 'Elmalı',
              'Ermenek', 'Etiler', 'Fener', 'Gebizli', 'Gençlik',
              'Güvenlik', 'Güzelbağ', 'Güzeloba', 'Güzeloluk', 'Haşimişcan',
              'Konuksever', 'Kırcami', 'Kızılarık', 'Kızılsaray', 'Kızıltoprak',
              'Kışla', 'Mehmetçik', 'Meltem', 'Memurevleri', 'Meydankavağı',
              'Muratpaşa', 'Yeşilbahçe', 'Selçuk', 'Sinan', 'Soğuksu',
              'Tahılpazarı', 'Topçular', 'Varlık', 'Yenigöl', 'Yenigün',
              'Sedir', 'Yeşildere', 'Yeşilova', 'Yüksekalan', 'Yıldız',
              'Zerdalilik', 'Çaybaşı', 'Çağlayan', 'Üçgen', 'Şirinyalı']
mahalle_kutu = Combobox(root, values = mahalleler)
mahalle_kutu.place(x = 100,y = 225)
mahalle_buton = tk.Button(root, text = "Seç", command = mahalle_duzenle, font="helvetica 12",borderwidth=6)
mahalle_buton.place(x = 100, y = 250)

# Isıtma Türü
isitma_label = tk.Label(text = "Isıtma Türü:", bg="#f0f8ff", font=("Helvetica", 12))
isitma_label.place(x=300, y=200)
isitmalar = ['Yerden Isıtma', 'Güneş Enerjisi', 'Şömine', 'Klima',
             'Kombi (Doğalgaz)', 'Merkezi (Pay Ölçer)', 'Kombi (Elektrik)',
             'Merkezi', 'Kat Kaloriferi', 'Fancoil Ünitesi',
             'Doğalgaz Sobası', 'Soba', 'Yok']
isitma_kutu = Combobox(root, values = isitmalar)
isitma_kutu.place(x = 300,y = 225)
isitma_buton = tk.Button(root, text = "Seç", command = isitma_duzenle, font="helvetica 12",borderwidth=6)
isitma_buton.place(x = 300, y = 250)

# Otopark Tipi
otopark_label = tk.Label(text = "Otopark Tipi:", bg="#f0f8ff", font=("Helvetica", 12))
otopark_label.place(x=500, y=200)
otoparklar = ['Açık & Kapalı Otopark', 'Kapalı Otopark', 'Açık Otopark', 'Yok']
otopark_kutu = Combobox(root, values = otoparklar)
otopark_kutu.place(x = 500,y = 225)
otopark_buton = tk.Button(root, text = "Seç", command = otopark_duzenle, font="helvetica 12",borderwidth=6)
otopark_buton.place(x = 500, y = 250)

# Eşya Durumu
esya_label = tk.Label(text = "Eşya Durumu:", bg="#f0f8ff", font=("Helvetica", 12))
esya_label.place(x=700, y=200)
esyalar = ['Eşyasız', 'Eşyalı']
esya_kutu = Combobox(root, values = esyalar)
esya_kutu.place(x = 700,y = 225)
esya_buton = tk.Button(root, text = "Seç", command = esya_durumu_duzenle, font="helvetica 12",borderwidth=6)
esya_buton.place(x = 700, y = 250)

# Sahibi
sahibi_label = tk.Label(text = "Sahibi:", bg="#f0f8ff", font=("Helvetica", 12))
sahibi_label.place(x=100, y=300)
sahipler = ['Bireysel', 'Emlakçı']
sahibi_kutu = Combobox(root, values = sahipler)
sahibi_kutu.place(x = 100,y = 325)
sahibi_buton = tk.Button(root, text = "Seç", command = sahibi_duzenle, font="helvetica 12",borderwidth=6)
sahibi_buton.place(x = 100, y = 350)

# Balkon Durumu
balkon_label = tk.Label(text = "Balkon Durumu:", bg="#f0f8ff", font=("Helvetica", 12))
balkon_label.place(x=300, y=300)
balkonlar = ['Yok', 'Var']
balkon_kutu = Combobox(root, values = balkonlar)
balkon_kutu.place(x = 300,y = 325)
balkon_buton = tk.Button(root, text = "Seç", command = balkon_duzenle, font="helvetica 12",borderwidth=6)
balkon_buton.place(x = 300, y = 350)

# Asansör Durumu
asansor_label = tk.Label(text = "Asansör Durumu:", bg="#f0f8ff", font=("Helvetica", 12))
asansor_label.place(x=500, y=300)
asansorler = ['Yok', 'Var']
asansor_kutu = Combobox(root, values = asansorler)
asansor_kutu.place(x = 500,y = 325)
asansor_buton = tk.Button(root, text = "Seç", command = asansor_duzenle, font="helvetica 12",borderwidth=6)
asansor_buton.place(x = 500, y = 350)

# Site Durumu
site_label = tk.Label(text = "Site Durumu:", bg="#f0f8ff", font=("Helvetica", 12))
site_label.place(x=700, y=300)
siteler = ['Site Dışında', 'Site İçinde']
site_kutu = Combobox(root, values = siteler)
site_kutu.place(x = 700,y = 325)
site_buton = tk.Button(root, text = "Seç", command = site_duzenle, font="helvetica 12",borderwidth=6)
site_buton.place(x = 700, y = 350)

# Dairenin Bulunduğu Kat
kat_label = tk.Label(root, text = "Dairenin Bulunduğu Kat:\n(Max. 18)", font="helvetica 12",borderwidth=6)
kat_label.place(x = 100, y = 400)
kat_entry = tk.Entry()
kat_entry.place(x = 100, y = 455)
kat_buton = tk.Button(root, text = "Seç", command = dairenin_kati_duzenle, font="helvetica 12",borderwidth=6)
kat_buton.place(x = 100, y = 480)

# Oda Sayısı
oda_label = tk.Label(root, text = "Oda Sayısı:\n(Max. 8)", font="helvetica 12",borderwidth=6)
oda_label.place(x = 300, y = 400)
oda_entry = tk.Entry()
oda_entry.place(x = 300, y = 455)
oda_buton = tk.Button(root, text = "Seç", command = oda_duzenle, font="helvetica 12",borderwidth=6)
oda_buton.place(x = 300, y = 480)

# Salon sayısı
salon_label = tk.Label(root, text = "Salon Sayısı:\n(Max. 3)", font="helvetica 12",borderwidth=6)
salon_label.place(x = 500, y = 400)
salon_entry = tk.Entry()
salon_entry.place(x = 500, y = 455)
salon_buton = tk.Button(root, text = "Seç", command = salon_duzenle, font="helvetica 12",borderwidth=6)
salon_buton.place(x = 500, y = 480)

# Depozito
depozito_label = tk.Label(root, text = "Depozito:\n(Max. 250bin)", font="helvetica 12",borderwidth=6)
depozito_label.place(x = 700, y = 400)
depozito_entry = tk.Entry()
depozito_entry.place(x = 700, y = 455)
depozito_buton = tk.Button(root, text = "Seç", command = depozito_duzenle, font="helvetica 12",borderwidth=6)
depozito_buton.place(x = 700, y = 480)

# Banyo sayısı
banyo_label = tk.Label(root, text = "Banyo Sayısı:\n(Max. 4)", font="helvetica 12",borderwidth=6)
banyo_label.place(x = 100, y = 530)
banyo_entry = tk.Entry()
banyo_entry.place(x = 100, y = 585)
banyo_buton = tk.Button(root, text = "Seç", command = banyo_duzenle, font="helvetica 12",borderwidth=6)
banyo_buton.place(x = 100, y = 610)

# Bina Kat Sayısı
bina_kat_label = tk.Label(root, text = "Bina Kat Sayısı:\n(Max. 18)", font="helvetica 12",borderwidth=6)
bina_kat_label.place(x = 300, y = 530)
bina_kat_entry = tk.Entry()
bina_kat_entry.place(x = 300, y = 585)
bina_kat_buton = tk.Button(root, text = "Seç", command = bina_kat_duzenle, font="helvetica 12",borderwidth=6)
bina_kat_buton.place(x = 300, y = 610)

# Bina Yaşı
yas_label = tk.Label(root, text = "Bina Yaşı:\n(Max. 45)", font="helvetica 12",borderwidth=6)
yas_label.place(x = 500, y = 530)
yas_entry = tk.Entry()
yas_entry.place(x = 500, y = 585)
yas_buton = tk.Button(root, text = "Seç", command = yas_duzenle, font="helvetica 12",borderwidth=6)
yas_buton.place(x = 500, y = 610)

# Net/Brüt Alan Oranı
alan_label = tk.Label(root, text = "Net/Brüt Alan Oranı:\n(Max. 1)", font="helvetica 12",borderwidth=6)
alan_label.place(x = 700, y = 530)
alan_entry = tk.Entry()
alan_entry.place(x = 700, y = 585)
alan_buton = tk.Button(root, text = "Seç", command = alan_duzenle, font="helvetica 12",borderwidth=6)
alan_buton.place(x = 700, y = 610)

def hesapla():
    yeni_veri = [[mahalle],[isitma],[otopark],[esya],[sahibi],[balkon],[asansor],[site],
                 [kat],[oda],[salon],[depozito],[banyo],[bina_kat],[yas],[alan]]  
    yeni_veri = pd.DataFrame(yeni_veri).T

    df3 = yeni_veri.rename(columns = {0:"mahalle",
                        1:"isitma_turu",
                        2:"otopark",
                        3:"esya_durumu",
                        4:"sahibi",
                        5:"balkon",
                        6:"asansor",
                        7:"site_icinde",
                        8:"dairenin_bulundugu_kat",
                        9:"oda_sayisi2",
                        10:"salon_sayisi",
                        11:"depozito",
                        12:"banyo_sayisi",
                        13:"bina_kat_sayisi",
                        14:"bina_yas",
                        15:"net_brut_orani",})
    pred = model_xgb.predict(df3)
    if(pred < 0):
        pred = -1*pred
    pred = int(pred)
    s2 = tk.Label(root, text = pred, font="helvetica 20",borderwidth=6, padx = 200, pady = 40)
    s2.place(x = 910, y = 550)

# Hesapla Butonu
hesapla_buton = tk.Button(root, text = "HESAPLA", command = hesapla, font="helvetica 15", borderwidth=50, padx = 50, pady = 50)
hesapla_buton.place(x = 900, y = 250)

s1 = tk.Label(root, text= "", font="helvetica 12",borderwidth=6, padx = 200, pady = 50)
s1.place(x = 910, y = 550)

root.mainloop()