# Proses Machine Learning Secara End To End


disc :
- upload file pada github dilakukan scr manual karena samapi saat ini masi terkendala pada saat push github.
- pengembangan maschine learning dilakukan end to end damun secara lokal, untuk pengembangan secara online masih terkendala.

## Pengembangan Model Machine Learning menggunakan Pada Notebook

### Data understanding dan preparation
Kualitas air adalah suatu ukuran kondisi air dilihat dari karakteristik fisik, kimiawi, dan biologisnya. Kualitas air juga menunjukkan ukuran kondisi air relatif terhadap kebutuhan biota air dan manusia. Kualitas air sering kali menjadi ukuran standar terhadap kondisi kesehatan ekosistem air dan kesehatan manusia terhadap air minum.
kali ini kita akan melakukan prediksi mengenai kelayakan air minum menggunakan machine learning, dan diperoleh data sbb :

<img src="assets/01. dataset.jpg" width="1000"/>

sc : kaggle

1. ph : adalah derajat keasaman yang digunakan untuk menyatakan tingkat keasaman atau kebasaan yang dimiliki oleh suatu larutan.
2. Hardness : istilah yang merujuk pada senyawa kandungan yang terlarut pada air itu sendiri, seperti magnesium dan ion kalsium sedangkan kandungan mineral air dipengaruhi dari bebatuan, sungai dan wilayah lain di mana air itu mengalir.
3. solids : atau Total Dissolved Solid (TDS) merupakan istilah untuk menandakan jumlah padatan terlarut atau konsentrasi jumlah ion kation (bermuatan positif) dan anion (bermuatan negatif) di dalam air
4. Chloramines : Bahan kimia chloramine terbentuk dari nitrogen--ditemukan dalam air seni manusia dan keringat--dengan klorin dalam kolam renang
5. Sulfate : Sulfat dan air bereaksi membentuk asam sulfat yang terbentuk melalui oksidasi. Sulfat berbahaya bagi kesehatan apabila telah melewati batas kadar maksimum.
6. Conductivity :  mengukur kemampuan air untuk melewatkan aliran listrik. Kemampuan ini langsung berhubungan ke konsentrasi ion di air.
7. Organic Carbon : Total Organic Carbon (TOC) merupakan ukuran tidak langsung dari molekul organik yang diukur sebagai karbon.
8. Trihalometahens : Trihalometan adalah senyawa yang terdiri dari empat bahan kimia, yaitu Kloroform, Bromodiklorometan, Dibromokolometan, dan Bromofrom. Senyawa ini terbentuk bersama dengan disinfeksi air yang dihasilkan dari klorin, apabila dikonsumsi dalam jumlah banyak dapat menyebabkan kanker kolon.
10. Turbidity : Kekeruhan dapat diartikan sebagai ukuran relatif kejernihan air. Kekeruhan bukanlah ukuran langsung dari partikel tersuspensi dalam air tetapi sebaliknya, yaitu ukuran efek hamburan partikel-partikel tersebut terhadap cahaya.

Setelah mengerti mengenai variabel data langkah selanjutnya dalam data analisis adalah :

### Cek Missing Data

<img src="assets/02. cek nan.jpg" width="500"/>

impute data

```
imputer = KNNImputer(n_neighbors=10, weights="uniform")
l=imputer.fit_transform(dataset)
waterpot=pd.DataFrame(l,columns=dataset.columns)
```

### Splitting Data
```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)
x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size = 0.5, random_state = 42, stratify = y_test)
```

Lalu dump data ke dalam pickle
```
utils.pickle_dump(dataset, config["dataset_cleaned_path"])
utils.pickle_dump(x_train, config["train_set_path"][0])
utils.pickle_dump(y_train, config["train_set_path"][1])
utils.pickle_dump(x_valid, config["valid_set_path"][0])
utils.pickle_dump(y_valid, config["valid_set_path"][1])
utils.pickle_dump(x_test, config["test_set_path"][0])
utils.pickle_dump(y_test, config["test_set_path"][1])
```
### Eksporatory Data Analysis

1. Lakukan ringkasan statistik pada data train

<img src="assets/03.deskriptif.jpg" width="500"/>

2. Lakukan pengecekan terhadap imbalance data

<img src="assets/06.targetbeforebalance.jpg" width="500"/>

3. Lakukan visualisasi Grafik dan Plot

<img src="assets/07.visualisasidata1.jpg" width="1000"/>
<img src="assets/07.visualisasidata2.jpg" width="1000"/>
<img src="assets/07.visualisasidata3.jpg" width="1000"/>

4. Lakukan perhitungan Korelasi antar variable 

<img src="assets/08.korelasi.jpg" width="500"/>

5. Lakukan Pengujian Statistik

<img src="assets/09.ujistatistik.jpg" width="500"/>\

### Data Preprocessing

Langkah yang perlu dilakukan pada preprocessing berdasarkan EDA adalah.
1. Oversample Data
```
x_ros, y_ros = SMOTE(random_state = 42).fit_resample(
    train_set.drop(columns = config["label"]),
    train_set[config["label"]]
)
train_set_bal = pd.concat([x_ros, y_ros], axis = 1)
```
<img src="assets/05.targetafterbalance.jpg" width="500"/>

2. Scaling data
```
scaler = MinMaxScaler()
names = x_ros.columns
d = scaler.fit_transform(x_ros)

x_std = pd.DataFrame(d, columns=names)
train_set_bal_std = pd.concat([x_std, y_ros], axis = 1)
```
hasilnya

<img src="assets/04.standardisasibfout.jpg" width="1000"/>

3. Remove Outlier

Data setelah di remove

<img src="assets/04.standardisasi.jpg" width="1000"/>

4. Dumping File Setelah preprocessing

```
utils.pickle_dump(train_set_bal_cleaned[config["predictors"]], config["train_feng_set_path"][0])
utils.pickle_dump(train_set_bal_cleaned[config["label"]], config["train_feng_set_path"][1])

utils.pickle_dump(valid_set[config["predictors"]], config["valid_feng_set_path"][0])
utils.pickle_dump(valid_set[config["label"]], config["valid_feng_set_path"][1])

utils.pickle_dump(test_set[config["predictors"]], config["test_feng_set_path"][0])
utils.pickle_dump(test_set[config["label"]], config["test_feng_set_path"][1])
```

### Modelling

Langkah dalam Model training
1. Pemodelan menggunakandata train
```
models = [RandomForestClassifier(), KNeighborsClassifier(), SVC(), LogisticRegression(),xgb.XGBClassifier()]
scores = dict()

for m in models:
    m.fit(x_train, y_train)
    y_pred = m.predict(x_test)

    print(f'model: {str(m)}')
    print(classification_report(y_test,y_pred, zero_division=1))
    print('-'*30, '\n')
 ```
 
hasilnya
 
<img src="assets/10.hasilmodelling.jpg" width="500"/>
 
 
2. Menghitung Confusion Matrix
 
<img src="assets/11.kofmat.jpg" width="500"/>
  
3. Parameter Tuning

```
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001], 
              'kernel': ['linear']} 
rfc_Model = SVC(gamma='auto',
                random_state = 123)
#Setup Random Grid Search
from sklearn.model_selection import RandomizedSearchCV 
rfc_RS = RandomizedSearchCV (estimator = rfc_Model,
                              param_distributions = param_grid,
                              n_iter = 10,
                              cv = 3, 
                              verbose=2, 
                              n_jobs = -1)
 ```
 
 hasil parameter Tuning
 
 <img src="assets/12.tuningparameter.jpg" width="500"/>


3. Dump Model dlm Pickle Data
```
utils.pickle_dump(rfc, config["production_model_path"])
model = utils.pickle_load("models/production_model.pkl")
```

### Convert script Notebook ke Python

Setelah selesai pembuatan model dalam notebook, selanjutnya  adalah mengkonversi script tersebut ke dalam python (.py) dengan cara membuat file baru kemudian membuat definisi tiap2 script

bentuk filenya dapat berusa sbb:

<img src="assets/15.scriptspython.jpg" width="200"/>

dari python tersebut kita coba jalankan atau dapat di tes menggunakan pytest, kemudian mendapatkan data berupa pickle yaang nantinya akan digunakan pada API dan Streamlit

<img src="assets/13.pickeldata.jpg" width="200"/>
<img src="assets/14.picklemodel.jpg" width="200"/>


### Buat API dan Streamlit

1. Buat script python API yang menerima inputan sbb
```
class api_data(BaseModel):
    ph : float
    Hardness : float
    Solids : float
    Chloramine : float
    Sulfate : float
    Conductivity : float
    Organic_carbon : float
    Trihalomethanes : float
    Turbidity : float
```
kemudian jalankan pada server lokal 
```
uvicorn.run("api:app", host = "0.0.0.0", port = 8080)
```

2. BuatScript Python streamlit yang mengirimkan data ke API sbb

```
**if submitted:
        # Create dict of all data in the form
        raw_data = {
            "ph": ph,
            "Hardness": hardness,
            "Solids": solids,
            "Chloramines": chloramines,
            "Sulfate": sulfate,
            "Conductivity": conductivity,
            "Organic_carbon": organic_carbon,
            "Trihalomethanes": trihalomethanes,
            "Turbidity": turbidity
```

3. Jalankan Streamlit lalu akan muncul pada browser sbb:

<img src="assets/16.streamlit.jpg" width="1000"/>

isikan form tersebut sesuai rangenya, kemudian klik submit maka pada API akan menerima data sbb:

<img src="assets/17.apiinput.jpg" width="300"/>

dan Outputnya

<img src="assets/18.apioutput.jpg" width="300"/>


### CATATAN

Selanjutnya bisa digunakan Docker, serta Server yang dapat melakukan prediksi secara Online.







