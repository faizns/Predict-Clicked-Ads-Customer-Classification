# 🖱 Predict Clicked Ads Customer Classification
<br>

**Tool** : Jupyter Notebook | [Link Notebook]()<br>
**Programming Language** : Python <br>
**Libraries** : Pandas, NumPy, Scikit Learn, shap <br>
**Visualization** : Matplotlib, Seaborn <br>
**Dataset** : Disediakan oleh Rakamin Academy - [Dataset]() <br>
<br>
<br>

**Table of Contents**
- [STAGE 0: Introduction]()
	- [Background]()
	- [Goals]()
    - [Objective]()
    - [Business Metric]()
- [STAGE 1: Exploratory Data Analysis]()
	- [Data Overview]()
	- [Data Quality Assessment]()
    - [Data Exploration]()
- [STAGE 2: Data Pre-processing]()
- [STAGE 3: Modeling]()
- [STAGE 4: Save Model]()
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

Tabel 1 - Deskripsi Fitur
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

Tabel 2 - Hasil Data Quality Assessment
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
    Gambar 2 — Plot korelasi Daily Time Spent on Site dengan Internet Usage terhadap Clicked on Ads
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

## 📂 **STAGE 3: Modeling**

<br>
<br>

---

## 📂 **STAGE 4: Business Recommendation**
