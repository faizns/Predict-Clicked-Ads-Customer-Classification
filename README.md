# 🖱 Predict Clicked Ads Customer Classification
<br>

**Tool** : Jupyter Notebook | [Link Notebook](https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification/blob/main/Customer%20Clicked%20Ads%20Classification.ipynb)<br>
**Programming Language** : Python <br>
**Libraries** : Pandas, NumPy, Scikit Learn, shap <br>
**Visualization** : Matplotlib, Seaborn <br>
**Source Dataset** : Rakamin Academy <br>
<br>
<br>

**Table of Contents**
- [STAGE 0: Introduction](https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification#-stage-0-introduction)
	- [Background](https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification#background)
	- [Goal](https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification#goal)
    - [Objective](https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification#objective)
    - [Business Metric](https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification#business-metric)
- [STAGE 1: Exploratory Data Analysis](https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification#-stage-1-exploratory-data-analysis)
	- [Data Overview](https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification#data-overview)
	- [Data Quality Assessment](https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification#data-quality-assesment)
    - [Data Exploration](https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification#data-exploration)
- [STAGE 2: Data Pre-processing](https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification#-stage-2-data-pre-processing)
- [STAGE 3: Data Modeling and Evaluation](https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification#-stage-3-data-modeling-and-evaluation)
	- [Model Experimet](https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification#model-experiment)
	- [Evaluation: Confussion Matrix](https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification#evaluation-confusion-matrix)
	- [Evaluation: Feature Importance](https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification#evaluation-feature-importance)
- [STAGE 4: Business Recommendation](https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification#-stage-4-business-recommendation)
<br>
<br>

---

## 📂 **STAGE 0: Introduction**
### Background
Seiring berkembangnya zaman, perusahaan harus mampu mengoptimalkan metode iklan mereka di platform digital untuk menarik calon pelanggan potensial dengan biaya yang minimal. Hal ini dilakukan dengan tujuan meningkatkan konversi, yaitu jumlah pelanggan potensial yang melakukan pembelian setelah mengklik iklan. Namun, untuk mencapai tujuan ini, perusahaan harus dapat melakukan prediksi click-through yang akurat. Click-through rate yang akurat sangat penting dalam menentukan keberhasilan kampanye iklan digital. Tanpa prediksi yang akurat, perusahaan mungkin akan mengeluarkan biaya yang besar tanpa hasil yang signifikan.<br>

### Goal
Membuat machine learning model yang dapat mendeteksi potential user untuk convert atau tertarik pada sebuah iklan, sehingga perusahaan bisa mengoptimalkan cost dalam beriklan. <br>

### Objective
- Memprediksi user yang memiliki potensi untuk klik iklan atau tidak dengan akurasi 90%
- Mendapatkan insight mengenai pola potensial user yang klik iklan
- Memberikan rekomendasi bisnis berdasarkan hasil analisis dan model <br>

### Business Metric
Click-through rate
<br>
<br>

---

## 📂 **STAGE 1: Exploratory Data Analysis**
### Data Overview
Dataset memiliki 1000 baris dan 9 fitur dengan 1 target. Berikut informasi fitur pada dataset:

Tabel 1 — Deskripsi Fitur
Fitur | Deskripsi
------|----------
Daily Time Spent on Site | Lamanya tinggal disuatu situs (harian) dalam satuan menit
Age | Umur user dalam satuan tahun
Area Income | Pendapatan user dalam satuan rupiah
Daily Internet Usage | Penggunaan internet harian dalam satuan menit
Male | Gender user
Timestamp | Kapan user visit sebuah situs
Clicked on Ad | Click atau tidak iklan yang ditampilkan
city | Kota asal user
province | Provinsi asal user
category | Kategori product yang dikunjungi
<br>

### Data Quality Assesment
Asesmen data dilakukan untuk memastikan bahwa data yang digunakan untuk analisis selanjutnya sudah siap dan sesuai dengan kebutuhan analisis. Hal yang dilakukan:
- Memeriksa missing value pada data
- Memeriksa duplikasi data
- Memeriksa tipe dan konsistensi nilai
- Memeriksa outlier atau data yang tidak biasa (anomali)

Tabel 2 — Hasil Data Quality Assessment
 **Data Assessment** | **Finding**  | **Handling** 
--------------------|--------------|--------------
Missing values | Terdapat missing value pada fitur `Daily Time Spent on Site`, `Area Income`, `Daily Internet Usage`, dan `Male` | Berdasarkan distribusi data missing value pata tipe data numerik diatasi dengan **imputasi** menggunakan nilai **median**. Sedangkan untuk fitur kategorikal, `Age` menggunakan **modus**.
Duplikat | **Tidak ada** duplikat data | Tidak dilakukan handling
Fitur atau nilai yang tidak sesuai | - Fitur yang tidak digunakan : `Unnamed: 0` <br> - Tipe data tidak sesuai : `Timestamp`| - Menghapus fitur `Unnamed: 0` dengan drop() <br> - Mengubah tipe data `Timestamp` dengan **datettime** dan dapat dilakukan ekstraksi bulan, minggu, hari, dan jam.
Anomali atau outlier | Fitur `Area Income` memiliki outlier, namun masih **dapat ditoleransi** karena bukan nilai yang ekstrim  | Tidak dilakukan handling
<br>

### Data Exploration
#### **Customer Type Behaviour Analysis on Advertisement**
Customer Type and Behaviour Analysis on Advertisement digunakan untuk **memahami profil pelanggan dan pola perilaku mereka terkait iklan**. Dalam analisis ini, data tentang demografi, kebiasaan, dan tanggapan pelanggan terhadap iklan dikumpulkan dan dianalisis secara mendalam. Fitur yang digunakan dalam analisis ini diantaranya adalah **Daily Internet Usage**, **Daily Time Spent**, dan **Age**.

Data Daily Internet Usage memberikan insight tentang sejauh apa pelanggan terlibat dalam aktivitas online. Informasi ini dapat untuk mengidentifikasi kelompok pelanggan yang cenderung lebih aktif secara online dan memanfaatkan internet dalam kehidupan sehari-hari mereka.

Selanjutnya, Daily Time Spent adalah data yang mencerminkan berapa lama pelanggan menghabiskan waktu mereka dalam aktivitas online setiap harinya. Informasi ini berguna dalam memahami sejauh mana pelanggan terlibat dalam konten digital dan seberapa besar potensi mereka untuk melihat atau berinteraksi dengan iklan.

Selain itu, faktor usia juga berperan penting dalam analisis ini. Usia dapat memberikan petunjuk tentang preferensi dan minat khusus dari kelompok pelanggan. Misalnya, generasi yang lebih muda mungkin lebih terbuka terhadap inovasi teknologi dan lebih aktif di media sosial, sedangkan generasi yang lebih tua mungkin lebih tertarik dengan konten yang relevan dengan kehidupan sehari-hari mereka.


<p align="center">
    <kbd> <img width="900" alt="Presentation1" src="https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification/assets/115857221/6544883a-4112-47d1-9b0a-db0fd503049c">
 </kbd> <br>
    Gambar 1 — Plot Distribusi Daily Internet Usage, Daily Time Spent, dan Age terhadap Clicked on Ads
</p>
<br>


Berdasarkan analisis dari plot Daily Time Spent, ditemukan bahwa **pengguna yang jarang menghabiskan waktu di sebuah situs (kurang dari 1 jam) memiliki potensi yang lebih besar untuk mengklik iklan**. Kemungkinan hal ini terjadi karena pengguna yang tidak terlalu lama berada di situs tersebut cenderung lebih terbuka untuk mengeksplorasi iklan yang muncul, dan mereka mungkin lebih mudah tergoda untuk mengeklik iklan tersebut.

Sementara itu, melalui analisis Daily Internet Usage, ditemukan bahwa **pengguna yang jarang menggunakan internet memiliki potensi yang lebih besar untuk mengklik iklan** dibandingkan dengan pengguna yang sering menggunakan internet. Pengguna yang jarang menggunakan internet mungkin memiliki rasa ingin tahu yang lebih besar tentang produk atau layanan yang ditawarkan melalui iklan. Karena mereka kurang terbiasa dengan internet, mereka mungkin merasa tertarik dengan iklan dan ingin mengetahui lebih banyak tentang produk tersebut. Selain itu, keterbatasan akses internet juga dapat menjadi faktor yang berperan, di mana pengguna yang jarang menggunakan internet akan lebih cenderung mengklik iklan yang menarik untuk mendapatkan informasi lebih lanjut.

Sementara dalam analisis usia, ditemukan bahwa **pengguna yang lebih tua memiliki potensi yang lebih besar untuk mengklik iklan**. Hal ini mungkin disebabkan oleh fakta bahwa pengguna internet yang lebih muda lebih terbiasa dengan teknologi dan internet, sehingga mereka mungkin lebih mampu mencari informasi melalui sumber lain selain iklan. Mereka juga cenderung lebih kritis dalam menilai iklan dan lebih memilih untuk menghindari iklan yang terlalu mengganggu atau tidak relevan. Di sisi lain, pengguna yang lebih tua mungkin memiliki ketertarikan yang lebih besar terhadap iklan yang relevan dengan kehidupan sehari-hari mereka, sehingga mereka lebih mungkin untuk mengklik iklan tersebut.

<p align="center">
    <kbd> <img> <img width="600" alt="korelasi" src="https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification/assets/115857221/6036e6b9-fe91-465b-bdcd-d6d1de8eb79e"></kbd> <br>
    Gambar 2 — Plot Korelasi Daily Time Spent on Site dengan Internet Usage terhadap Clicked on Ads
</p>
<br>

Dari plot korelasi antara Daily Time Spent on Site dengan Internet Usage terhadap Target, ditemukan bahwa distribusi pengguna dapat dibagi menjadi dua segmen yaitu pengguna aktif dan non-aktif. **Pengguna aktif cenderung menghabiskan lebih banyak waktu di situs dan lebih terlibat dalam penggunaan internet secara keseluruhan**. Namun, menariknya, **pengguna aktif cenderung tidak terlalu suka mengklik iklan yang ditampilkan**.

Berdasarkan temuan ini, perusahaan dapat mengoptimalkan sistem iklannya dengan memfokuskan target kepada pengguna non-aktif. Pengguna non-aktif ini mungkin memiliki waktu yang lebih sedikit dihabiskan di situs dan penggunaan internet secara umum. Oleh karena itu, mereka mungkin lebih rentan terhadap iklan dan memiliki kemungkinan yang lebih tinggi untuk mengklik iklan yang ditampilkan. Dengan memfokuskan strategi iklan kepada pengguna non-aktif, perusahaan dapat meningkatkan efektivitas kampanye iklannya. Hal ini dapat dilakukan dengan menyesuaikan konten iklan agar lebih menarik dan relevan bagi pengguna non-aktif serta memilih situs pemasaran yang tepat untuk mencapai mereka. Dengan melakukan pendekatan yang lebih spesifik terhadap pengguna non-aktif, perusahaan dapat mengoptimalkan sistem advertisement-nya dan meningkatkan peluang untuk mendapatkan respons yang lebih baik dari target audiens yang dituju.<br>
<br>


#### **Time Analysis of User Clicks on Ads**

Time Analysis of User Clicks on Ads digunakan untuk **menganalisis pola waktu pengguna saat mengklik iklan dengan mengidentifikasi tren dan pola yang dapat memberikan insight**. Dengan analisis ini, perusahaan dapat menentukan jam-jam atau periode tertentu di mana pengguna cenderung lebih aktif dalam mengklik iklan dan mengoptimalkan strategi penempatannya. Hal tersebut dilakukan agar dapat mencapai audiens yang lebih responsif dan meningkatkan peluang untuk mendapatkan klik yang lebih banyak. Selain itu, analisis ini juga dapat membantu perusahaan untuk mengalokasikan anggaran iklan dengan lebih efisien dengan mengarahkan sumber daya ke periode waktu yang paling menguntungkan.

<p align="center">
    <kbd> <img width="600" alt="Weekday" src="https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification/assets/115857221/f4805cbd-e118-4087-90b6-286c6cc63e95"></kbd> <br>
    Gambar 3 — Tren Harian Clicked on Ads
</p>
<br>

Dalam analisis mengenai perilaku pengguna terhadap klik iklan pada hari-hari tertentu, **sedikit sekali pengguna yang mengklik iklan pada hari Senin dan Jumat**. Hari Senin dan Jumat sering dianggap sebagai awal dan akhir minggu kerja, di mana orang cenderung sibuk dengan pekerjaan dan memiliki konsentrasi yang lebih rendah pada aktivitas online seperti mengklik iklan. Faktor ini dapat menjelaskan mengapa jumlah pengguna yang mengklik iklan pada hari-hari ini cenderung rendah.

Namun, **pada hari Rabu terlihat adanya konversi klik iklan yang paling baik**. Pada hari ini, jumlah pengguna yang mengklik iklan relatif tinggi, sementara jumlah pengguna yang tidak mengklik iklan rendah. Hal ini mungkin disebabkan oleh fakta bahwa hari Rabu sering dianggap sebagai titik tengah minggu di mana orang merasa lebih rileks dan memiliki lebih banyak waktu untuk menghabiskan waktu online serta melakukan aktivitas seperti berbelanja.

Selain itu, terdapat data menarik bahwa hari **Selasa dan Sabtu memiliki tingkat lalu lintas yang tinggi dan sekitar 50% pengguna cenderung mengklik iklan**. Ini menunjukkan bahwa pada hari-hari ini, ada peluang yang cukup baik untuk mencapai pengguna dengan iklan yang relevan.

<p align="center">
    <kbd> <img width="600" alt="hour" src="https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification/assets/115857221/12b86319-855f-421b-9645-dcfbfe204d24"></kbd> <br>
    Gambar 4 — Tren Setiap Jam Clicked on Ads
</p>
<br>

Analisis berdasarkan waktu jam menunjukkan bahwa terdapat **potensi pengguna untuk mengklik iklan dan memiliki konversi pembelian yang tinggi pada jam-jam tertentu, yaitu pukul 00.00, 09.00, 11.00, dan 18.00**.

Pada jam 00.00, dapat diasumsikan bahwa orang-orang cenderung tidak memiliki kewajiban atau tugas yang mendesak, sehingga lebih mungkin untuk menghabiskan waktu online dan menjelajahi internet. Kondisi ini menciptakan peluang bagi perusahaan untuk menampilkan iklan yang menarik pada jam ini dan meningkatkan kemungkinan pengguna mengklik iklan tersebut.

Jam 09.00 dan 11.00 mungkin merupakan waktu di mana orang memiliki jeda dalam pekerjaan atau mengambil istirahat singkat. Selama periode ini, pengguna mungkin memanfaatkan waktu luang mereka dengan menggunakan perangkat digital, seperti ponsel atau komputer, dan menjelajahi konten online. Oleh karena itu, menampilkan iklan yang relevan dan menarik pada jam-jam ini dapat meningkatkan potensi pengguna untuk mengklik iklan dan bahkan melakukan pembelian.

Sementara itu, pada jam 18.00, setelah selesai bekerja, pengguna dapat lebih fokus pada kegiatan pribadi dan bersantai. Pada saat ini, mereka cenderung memiliki lebih banyak waktu luang untuk menggunakan perangkat digital dan menjelajahi internet. Menggunakan jam ini sebagai waktu untuk menampilkan iklan dapat memberikan peluang yang baik untuk mencapai target audiens yang lebih responsif dan mempengaruhi mereka untuk melakukan tindakan, seperti mengklik iklan dan melakukan pembelian.

<br>
<br>

---
## 📂 **STAGE 2: Data Pre-processing**

Berikut tahapan-tahapan dalam Data Pre-processing yang telah dilakukan.

<p align="center">
    <kbd> <img width="800" alt="fitur" src="https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification/assets/115857221/3214a70a-1274-4f53-a4d2-18041c5c301f"> </kbd> <br>
    Gambar 5 — Tahap Data Pre-processing
</p>
<br>

Fitur yang digunakan untuk model.

<p align="center">
    <kbd> <img width="400" alt="fitur" src="https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification/assets/115857221/b05971a5-e16e-4004-8173-7ff5469d7148"></kbd> <br>
    Gambar 6 — Fitur yang digunakan untuk Model
</p>

<br>
<br>

---

## 📂 **STAGE 3: Data Modeling and Evaluation**
### **Model Experiment**
Untuk melakukan prediksi pada klik iklan, dilakukan dua eksperimen yang berbeda. Pada eksperimen pertama, data train default digunakan untuk melatih model. Eksperimen ini memanfaatkan data train dalam bentuk default atau tanpa adanya penyesuaian tambahan. Sementara itu pada eksperimen kedua, data distandardisasi menggunakan SandardScaler. Hal ini dikarenakan distribusi data cenderung mendekati normal, sehingga perlu dilakukan standardisasi agar data memiliki skala yang serupa.

Dalam kedua eksperimen ini, matriks akurasi digunakan sebagai metrik evaluasi. Matriks akurasi memberikan gambaran tentang seberapa baik model dapat mengklasifikasikan data dengan benar. Penggunaan matriks akurasi ini dipilih karena jumlah kategori pada target (Clicked on Ads) yang digunakan dalam analisis seimbang, yaitu memiliki jumlah pengguna yang mengklik iklan dan tidak mengklik yang relatif setara.

<p align="center">
    Tabel 3 — Hasil Eksperimen Pertama (Tanpa Standardization) <br>
    <kbd><img width="500" alt="ex1" src="https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification/assets/115857221/bddc03e7-cb76-49ad-8acf-2f401f20ab32"></kbd> <br>
</p>

<p align="center">
    Tabel 4 — Hasil Eksperimen Kedua (Standardization) <br>
    <kbd> <img width="500" alt="ex2" src="https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification/assets/115857221/e31f61bc-3cda-48cd-a556-25bb7cba4a3d"></kbd> <br>
</p>
<br>

Pada hasil eksperimen, terlihat bahwa algoritma **Random Forest memiliki akurasi tertinggi baik pada eksperimen pertama maupun kedua, dengan nilai akurasi mencapai 96%**. Selain itu, algoritma-algoritma lain seperti Gradient Boosting, XGBoost, dan LGBM juga menunjukkan akurasi yang tinggi pada eksperimen pertama, dengan nilai akurasi sebesar 95%. Pada eksperimen kedua, ketiga algoritma tersebut juga memberikan hasil akurasi yang hampir sama. Menariknya, terlihat bahwa penggunaan metode standardization tidak memberikan perubahan yang signifikan pada nilai akurasi untuk algoritma-algoritma tersebut. Hal ini mengindikasikan bahwa model tidak terlalu sensitif terhadap perbedaan skala fitur dalam data. Dengan kata lain, perbedaan skala fitur tidak memiliki pengaruh yang signifikan pada kinerja model. 

Selain itu algoritma seperti Random Forest, XGBoost, Gradient Boosting, dan LGBM termasuk dalam kategori algoritma yang robust dan memiliki kemampuan yang kuat dalam menangani berbagai jenis data. Mereka dapat menyesuaikan dengan baik terhadap data yang tidak distandardisasi, sehingga tidak memerlukan proses preprocessing yang rumit. Oleh karena itu, nilai akurasi mereka tidak banyak berubah ketika fitur-fitur distandardisasi atau tidak distandardisasi. <br>
<br>

### **Evaluation: Confusion Matrix**

<p align="center">
    <kbd><img width="500" alt="CM" src="https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification/assets/115857221/c64e7a95-78c7-401d-9404-345ed06afbac"></kbd> <br>
    Gambar 7 — Confussion Matrix Random Forest
</p>

Berdasarkan model random forest, performa model secara mendetail dievaluasi menggunakan confusion matrix. Berdasarkan hasil confusion matrix, model Random Forest menunjukkan **performa yang sangat baik** dalam memprediksi pengguna yang mengklik iklan atau tidak. Nilai kesalahan prediksi, yang terdiri dari False Positive (prediksi salah bahwa pengguna mengklik iklan) dan False Negative (prediksi salah bahwa pengguna tidak mengklik iklan), memiliki jumlah yang kecil. **Kesalahan prediksi yang kecil ini dapat dianggap sebagai prediksi yang akurat**. Berdasarkan hal tersebut, perusahaan dapat menggunakan model ini untuk mengidentifikasi dengan akurat pengguna yang memiliki potensi untuk mengklik iklan, sehingga dapat mengoptimalkan strategi pemasarannya. <br>
<br>

#### **Evaluation: Feature Importance**
<p align="center">
    <kbd> <img width="700" alt="SHAP" src="https://github.com/faizns/Predict-Clicked-Ads-Customer-Classification/assets/115857221/bef0d0a4-27eb-4dc9-aabd-d2cc8e05d8d7"> </kbd> <br>
    Gambar 8 — Feature Importance Random Forest
</p>

Analisis Feature Importance digunakan untuk mengidentifikasi fitur yang paling penting dalam membangun model. Dalam analisis menggunakan plot SHAP, dapat terlihat fitur yang memiliki pengaruh tertinggi terhadap prediksi klik pada iklan. Beberapa fitur yang menunjukkan pengaruh yang signifikan adalah Daily Internet Usage, Daily Time Spent on Site, Area Income, dan Age. Fitur seperti **Daily Internet Usage, Daily Time Spent on Site, dan Area Income memiliki korelasi negatif** terhadap klik iklan yang ditandai dengan warna merah pada sisi kiri plot. Hal ini menunjukkan bahwa pengguna yang memiliki kebiasaan penggunaan internet yang kurang aktif dan pengguna dengan pendapatan menengah ke bawah memiliki potensi yang lebih tinggi untuk mengklik iklan. Di sisi lain, fitur **Age memiliki korelasi yang positif** dengan klik iklan. Artinya, semakin tua usia pengguna, semakin tinggi potensi mereka untuk mengklik iklan yang ditampilkan.

Informasi mengenai Feature Importance ini dapat digunakan untuk mengoptimalkan strategi pemasaran dan menyusun iklan yang lebih efektif dengan mempertimbangkan karakteristik user berdasarkan fitur-fitur yang memiliki pengaruh signifikan dalam model.

<br>
<br>

---

## 📂 **STAGE 4: Business Recommendation**

Rekomendasi berdasarkan Feature Importance dan insight yang telah ditemukan.
- Perusahaan dapat **mentargetkan iklan pada pengguna internet non-aktif**, yaitu pengguna yang jarang menghabiskan waktu di situs (kurang dari 1 jam) dan pengguna yang jarang menggunakan internet (dengan Daily Internet Usage di bawah 2,5 jam sehari). Strategi yang dapat dilakukan diantaranya adalah :
    - Karena pengguna non-aktif memiliki keterbatasan waktu, penting untuk menciptakan iklan yang singkat dan menarik.Menggunakan pesan yang padat dan jelas dengan pemilihan kata yang tepat dapat menarik perhatian mereka dalam waktu singkat. 
    - Memanfaatkan strategi retargeting untuk terus berkomunikasi dengan pengguna non-aktif. Setelah mereka mengklik iklan awal dan menunjukkan minat, tampilkan iklan yang relevan secara berulang kali di berbagai platform yang mereka kunjungi, seperti situs web lain, aplikasi, atau media sosial. Ini dapat membantu meningkatkan awareness pengguna.
    - Konten iklan Anda relevan dengan minat dan kebutuhan pengguna non-aktif.<br>

- Perusahaan dapat mengarahkan strategi pemasaran dan iklan pada **segmen pasar kelompok usia lebih dari 40 tahun**. Strategi yang dapat dilakukan diantaranya adalah :
    - Memfokuskan kampanye iklan yang memiliki dampak atau relevansi dengan kehidupan dan kebutuhan kelompok usia di atas 40 tahun.
    - Desain iklan yang mudah dibaca dan sederhana oleh kelompok usia di atas 40 tahun.
    - Menggunakan platform iklan yang sesuai, seperti Facebook. Kelompok usia di atas 40 tahun cenderung lebih sedikit terlibat dalam media sosial dibandingkan dengan kelompok usia yang lebih muda. <br>

- Perusahaan dapat mengarahkan strategi pemasaran dan iklan pada **segmen pasar kelompok dengan pendapatan menengah kebawah (< 400juta/tahun)**. Strategi yang dapat dilakukan diantaranya adalah :
    - Memberikan iklan dengan penawaran harga yang terjangkau dan sesuai dengan anggaran pengguna dalam kisaran, seperi diskon khusus, bundel, atau harga promo untuk mendorong mereka untuk mengklik iklan. <br>

- Perusahaan dapat **memanfaatkan hari Rabu dan tingkat lalu lintas tinggi pada Selasa dan Sabtu**. Hari Rabu menunjukkan konversi klik iklan yang baik, sementara Selasa dan Sabtu memiliki tingkat lalu lintas yang tinggi dengan sekitar 50% pengguna cenderung mengklik iklan. Hal tersebut dapat digunakan untuk memaksimalkan penayangan iklan.

- Perusahaan dapat menggunakan **jam-jam yang berpotensi klik iklan** dengan memastikan penayangan iklan yang tepat pada saat-saat tersebut. Jam-jam pukul **00.00, 09.00, 11.00, dan 18.00** menunjukkan potensi pengguna untuk mengklik iklan dan memiliki konversi pembelian yang tinggi.

-  Apabila perusahaan ingin menargetkan kelompok pengguna aktif, strategi iklan dengan pendekatan softselling dapat menjadi pilihan yang efektif. Dalam strategi ini, perusahaan dapat fokus pada membangun hubungan yang baik dengan calon konsumen, memberikan informasi yang bermanfaat tentang produk atau layanan, dan menyoroti nilai-nilai yang dimiliki. Selain itu, pemilihan platform media sosial sebagai platform penayangan iklan juga dapat efektif, mengingat kelompok pengguna aktif cenderung menggunakan media sosial secara intensif. 


