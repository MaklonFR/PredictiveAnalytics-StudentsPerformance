# Laporan Proyek Machine Learning - Maklon Jacob Frare

## Domain Proyek

Kinerja siswa dalam lingkungan pendidikan memainkan peran penting dalam menentukan keberhasilan akademik dan masa depan mereka. Dalam era digital saat ini, institusi pendidikan memiliki akses ke berbagai data terkait aktivitas belajar siswa, seperti nilai ujian, kehadiran, partisipasi dalam kegiatan ekstrakurikuler, dan data demografis. Namun, potensi data ini seringkali belum dimanfaatkan secara maksimal untuk memberikan wawasan yang dapat membantu meningkatkan hasil belajar siswa.

Pendekatan tradisional dalam mengevaluasi kinerja siswa cenderung bersifat reaktif, hanya memberikan perhatian setelah masalah terjadi, seperti penurunan nilai atau tingkat kehadiran yang rendah. Oleh karena itu, diperlukan solusi yang bersifat prediktif untuk mengidentifikasi potensi risiko lebih awal, sehingga institusi pendidikan dapat mengambil langkah-langkah preventif untuk mendukung siswa secara proaktif.

Predictive analytics memungkinkan institusi pendidikan untuk mengidentifikasi pola performa siswa berdasarkan data historis. Teknologi ini membantu dalam pengambilan keputusan yang proaktif untuk memberikan intervensi yang tepat waktu dan mendukung siswa dalam mencapai hasil terbaik. Pada kasus ini penulis menerapkan 4 model pembelajaran machine learning yakni Random Forest, K-Nearest-Neighboars (KNN), Support Vector Machine (SVM) dan Extreme Gradient Boosting (XGBoost). Pendekatan ini mengintegrasikan keunggulan dari berbagai model untuk membandingkan dan menemukan algoritma terbaik dalam memprediksi performa siswa berdasarkan dataset yang diperoleh dari kaggel dan dapat diakses pada link [berikut](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset). Dengan menggunakan model ini, diharapkan hasil prediksi yang akurat dapat membantu institusi pendidikan dalam merancang strategi pembelajaran yang lebih efektif dan personalisasi untuk siswa.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang tersebut, maka rincian permasalahan yang dapat dibahas pada proyek ini yakni:
1. Berapa banyak waktu belajar mingguan (*Study Time Weekly*) yang optimal untuk meningkatkan GPA siswa?
2. Apakah absensi siswa (*Absences*) berkorelasi negatif dengan GPA mereka?
3. Bagaimana tutoring (bimbingan belajar) memengaruhi GPA siswa?
4. Apakah terdapat perbedaan performa akademik antara siswa laki-laki dan perempuan (*Gender*) dalam hal GPA?
5. Bagaimana partisipasi siswa dalam kegiatan ekstrakurikuler (*Extracurricular, Sports, Music, Volunteering*) memengaruhi GPA mereka?
6. Apakah dukungan orang tua (*Parental Support*) berhubungan langsung dengan GPA siswa?
7. Faktor mana yang paling berpengaruh terhadap prediksi GPA siswa ketika mempertimbangkan semua atribut (*Age, Gender, Parental Education, Study Time Weekly, Absences, Tutoring, Parental Support, Extracurricular, Sports, Music, Volunteering*)
8. Apa model terbaik yang dapat digunakan untuk memprediksi kinerja siswa?

### Goals
Berdasarkan problem statements, berikut tujuan yang ingin dicapai pada proyek ini.
1. Menampilkan durasi belajar yang lebih efektif.
2. Mendeteksi pola absensi yang berdampak pada penurunan performa akademik.
3. Menilai efektivitas bimbingan belajar untuk meningkatkan performa siswa.
4. Mengidentifikasi apakah ada kesenjangan gender dalam pencapaian akademik.
5. Menilai dampak kegiatan non-akademik terhadap kinerja akademik.
6. Mengukur pentingnya keterlibatan orang tua dalam keberhasilan belajar siswa.
7. Membangun model prediksi untuk memantau dan meningkatkan kinerja siswa
8. Menemukan model terbaik berdasarkan akurasi tertinggi untuk memprediksi kinerja siswa.

### Solution Statement
1. Melakukan proses *Exploratory Data Analysis* (EDA) untuk menampilkan durasi belajar yang lebih efektif, mendeteksi pola absensi yang berdampak pada penurunan performa akademik, menilai efektivitas bimbingan belajar untuk meningkatkan performa siswa, mengidentifikasi apakah ada kesenjangan gender dalam pencapaian akademik, menilai dampak kegiatan non-akademik terhadap kinerja akademik, mengukur pentingnya keterlibatan orang tua dalam keberhasilan belajar siswa, membangun model prediksi untuk memantau dan meningkatkan kinerja siswa, memahami sinergi antara bimbingan belajar dan durasi belajar mandiri, menemukan model terbaik berdasarkan akurasi tertinggi untuk memprediksi kinerja siswa.
2. Menggunakan 4 model *machine learning* yaitu *Extreme Gradient Boosting* (XGBoost), *Support Vector Machine* (SVM), *Decision Tree* (Tree), dan *Random Forest* untuk memprediksi kinerja siswa
3. Menggunakan confusion matrix dan f1 score pada masing-masing model *machine learning* untuk menemukan model terbaik berdasarkan akurasi tertinggi.

## Data Understanding
Dataset yang digunakan untuk mempredisksi kinerja siswa diambil dari platform [kaggle](https://www.kaggle.com/) yang dipublikasikan oleh Rabie El Kharoua pada tanggal 13 Juni 2024. Kumpulan data ini berisi informasi lengkap tentang 2.392 siswa sekolah menengah, yang merinci demografi, kebiasaan belajar, keterlibatan orang tua, kegiatan ekstrakurikuler, dan prestasi akademik mereka. Variabel target, GradeClass, mengklasifikasikan nilai siswa ke dalam kategori yang berbeda, sehingga menyediakan kumpulan data yang kuat untuk penelitian pendidikan, pemodelan prediktif, dan analisis statistik. Dataset ini terdiri dari 1 file csv.<br>

### Informasi Keterangan Variabel pada Data
Dataset ini memiliki 15 variabel dengan keterangan sebagai berikut.
Variabel | Keterangan
----------|----------
StudentID | Pengidentifikasi unik yang diberikan kepada setiap siswa (1001 hingga 3392)
Age | Usia siswa berkisar antara 15 hingga 18 tahun
Gender | Jenis kelamin siswa, di mana 0 mewakili Laki-laki dan 1 mewakili Perempuan
Ethnicity | Etnis siswa, dikodekan sebagai berikut: 0(Kaukasia), 1(Afrika Amerika), 2(Asia), 3(Lainnya)
ParentalEducation | Tingkat pendidikan orang tua, dikodekan sebagai berikut: 0(Tidak Ada), 1(Sekolah Menengah Atas), 2(Beberapa Perguruan Tinggi), 3(Sarjana), 4(Lebih Tinggi)
StudyTimeWeekly | Waktu belajar mingguan dalam jam, berkisar antara 0 hingga 20
Absences | Jumlah ketidakhadiran selama tahun ajaran, berkisar antara 0 hingga 30
Tutoring | Status bimbingan belajar, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya
ParentalSupport | Tingkat dukungan orang tua, dikodekan sebagai berikut: 0(Tidak Ada), 1(Rendah), 2(Sedang), 3(Tinggi), 4(Sangat Tinggi)
Extracurricular | Partisipasi dalam kegiatan ekstrakurikuler, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya
Sports | Partisipasi dalam olahraga, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya
Music	| Partisipasi dalam kegiatan musik, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya
Volunteering	| Partisipasi dalam kesukarelaan, di mana 0 menunjukkan Tidak dan 1 menunjukkan Ya
GPA	| Nilai Rata-rata Poin pada skala 2,0 hingga 4,0
GradeClass | Klasifikasi nilai siswa berdasarkan IPK (0: 'A' (IPK >= 3,5)), (1: 'B' (3,0 <= IPK < 3,5)), (2: 'C' (2,5 <= IPK < 3,0)), (3: 'D' (2,0 <= IPK < 2,5)), (4: 'F' (IPK < 2,0))

### Data Cleaning
Setelah diperiksa apakah terdapat kolom yang bernilai null, hasilnya adalah tidak ada. Sedangkan data duplikat atau data ganda juga tidak ada. Maka dengan demikian data siapa untuk dianalisis pada tahap selanjutnya.

Selanjutnya kita akan melihat pesebaran data pada kolom nuerikal pada gambar dibawah ini:

<img src = "images/data-numerikal.png"/> <br>
Interprestasi:
1. Pada kolom Age, dapat dilihat rata-rata siswa berumur 15-17 tahun. Dapat disimpulkan tidak ada Outlier yang tersebar.
2. Pada kolom StudyTimeWeekly, dapat dilihat bahwa rata-rata siswa memiliki waktu belajar 5-14 jam per minggu.
3. Pada kolom absences, dapat dilihat bahwa rata-rata siswa memiliki jumlah ketidakhadiran 6 - 23 hari. Dapat disimpulkan 1. Pada kolom
4. Pada kolom GPA, dapat dilihat bahwa rata-rata prestasi siswa diantara 1,2 - 2,7 dan tidak memiliki outlier.

### Univariate Analysis

Dari variabel-variabel yang diketahui, variabel dapat dibagi menjadi 2 jenis, yaitu variabel numerikal dan variabel kategorikal. Berikut merupakan kolom-kolom yang termasuk dalam variabel numerikal maupun kategorikal. <br>
Semua numerikal: ["Age", "StudyTimeWeekly", "Absences", "GPA"] <br>
Semua kategorikal: ["Gender", "Ethnicity", "ParentalEducation", "Tutoring", "ParentalSupport", "Extracurricular", "Sports", "Music", "Volunteering", "GradeClass"]

Pertama, kita akan melihat nilai berbeda pada kolom kategorikal pada gambar tabel dibawah ini:

<img src = "images/nilai-beda-data-katergorikal.png"/> <br>
Dapat dilihat pada tabel nilai berbeda pada:
1. Kolom gender = 2;
2. Kolom Etnicity = 4;
3. Kolom ParentalEducation = 5;
4. Kolom Tutoring = 2;
5. Kolom ParentalSupport = 5;
6. Kolom Extracurricular = 2;
7. Kolom Sports = 2;
8. Kolom Music = 2;
9. Kolom Volunteering = 2;
10. Kolom GradeClass(Variabel Target) = 5;
    
Kedua, kita akan memvisualisasikan kolom-kolom kategorikal untuk melihat jumlah-jumlah nilai kategorikal menggunakan bar plot.

<img src = "images/nilai-nilai-data-kategorikal-1.png"/> <br>
<img src = "images/nilai-nilai-data-kategorikal-2.png"/> <br>

Interpretasi:
1. Grafik jenis kelamin, menunjukan jumlah merata antara laki-laki dan perempuan.
2. Grafik Etnis, menunjukan mayoritas siswa berasal dari etnis kaukasia.
3. Grafisk pendidikan orangtua, menunjukan mayoritas pendidikan orang tua yakni pensisikan tinggi dan sarjana
4. Grafik bimbingan belajar, menunjukan mayoritas siswa tidak mengikuti bimbingan belajar.
5. Grafik dukungan orang tua, menunjukan mayoritas dukungan orang tua berada di level sdang dan tinggi.
6. Grafik Ekstrakulikuler(EKtrakulikuler, Olahraga, Musik dan Sukrelawan), menujukan rendahnya minat siswa pada kegiatan diluar sekolah.

Ketiga, kita akan melihat lebih detail mengenai jumlah dari masing-masing tingkat kelas terbaik yang menjadi target kita untuk mengetahui jumlah secara umum.

<img src = "images/daftar-nilai-prestasi-kelas.png"/> <br>

Interpretasi: 
Kelas terbaik(GradeClass) yang ditampilkan menununjukan mayoritas prestasi siswa berada di kategori Grade F (Prestasi terendah)

Keempat, kita buat pie chart untuk melihat persebaran data dari masing-masing Prestasi Siswa Pada Kategori GradeClass:

<img src = "images/pesebaran-gradeclass.png"/> <br>
Interpretasi:
Siswa berada pada gradeclass F (Prestasi terendah) memiliki presntasi terbayak yakni 50% keatas.

Langkah terakhir, kita akan membentuk histogram dari variabel-variabel numerikal untuk melihat persebaran data:

<img src = "images/pesebaran-data-numerikal.png"/> <br>
Interpretasi: 
Usia, waktu belajar setiap minggu, absen dan nilai siswa cukup berdistribusi normal.

### Multivariate Analysis

#### 1. Membandingkan waktu belajar mingguan dengan prestasi siswa

<img src = "images/pengaruh-waktu-belajar-prestasi-siswa"/> <br>

Interpretasi:
Siswa yang waktu belajaranya banyak mempengaruhi naiknya prestasi belajar(GPA).

#### 2. Membandingkan Ketidakhadiran(Absen) dengan Nilai Prestasi Siswa

<img src = "images/pengaruh-absen-prestasi-siswa.png"/> <br>

Interpretasi:
Absen(ketidakhadiran) siswa sangat mempengaruhi turun prestasinya(GPA).

#### 3. Membandingkan Bimbingan Belajar dengan Nilai Prestasi Siswa

<img src = "images/bimbel-prestasi-siswa.png"/> <br>

Interpretasi:
Banyak siswa yang tidak mengikuti bimbingan belajar yang mendapat prestasi rendah (Grade F)

#### 4. Membandingkan Jenis Kelamin Pada Nilai Prestasi Siswa

<img src = "images/jk-prestasi-siswa.png"/> <br>

Interpretasi:
Jenis kelamin pria lebih dominan memiliki prestasi lebih tinggi dibandingkan dengan wanita

#### 5. Membandingkan Kegiatan Ekskulikuler dengan Nilai Prestasi Siswa

<img src = "images/ekstra-prestasi-siswa.png"/> <br>

Interpretasi:
Lebih banyak siswa yang tidak mengikuti kegiatan ekstrakulikuler, olahraga dan musik mempengaruhi turunya nilai pretasi(GPA) mereka

#### 6. Membandingkan Dukungan Orang Tua dengan Nilai Prestasi Siswa

<img src = "images/dukungan-ortu-prestasi-siswa.png"/> <br>

Interpretasi:
Mayoritas dukungan orang tua sangat mempengaruhin nilai prestasi siswa (GPA). Semakin tinggi dukungan orang tua, maka semakin meningkat nilai prestasi dari anaknya.

#### 7. Melihat Korelasi Variabel dengan Menggunakan Heatmap

<img src = "images/korelasi-headmap.png"/> <br>

Interpretasi:

Nilai Prestasi Siswa memiliki
1. Korelasi negatif yang cukup kuat dengan ketidakhadiran(Absences).
2. Korelasi positif yang cukup lemah dengan waktu belajar setiap minggu(StudyTimeWeekly).

#### 8. Melihat Plot Scatter yang Memiliki Nilai Korelasi Positif dan Negatif

<img src = "images/plot-scatter-absen-study-timeweekly-gpa.png"/> <br>

Interpretasi:

Nilai prestasi siswa (GPA) memiliki  korelasi negatif yang kuat pada ketidakhadiran (garis regresi menurun ke kanan bawah) dan korelatif positif cukup lemah pada waktu belajar setiap minggu (garis regresi naik ke kanan atas)

## Data Preparation
Pertama, akan diubah nilai-nilai kategorikal pada data menggunakan encoder sehingga menjadi nilai-nilai numerik agar dapat dilatih dengan *machine learning*.

### Encoding Kategorikal

Encoding Kategorikal dilakukan terhadap 4 variabel, yaitu
* `family_history_with_overweight` (Apakah terdapat anggota keluarga responden yang juga terkena obesitas)
* `FAVC` (Apakah responden mengonsumsi makanan berkalori tinggi)
* `SMOKE` (Apakah responden merupakan perokok)
* `SCC` (Apakah responden memantau asupan kalori)

karena nilai-nilai pada keempat variabel tersebut hanya `yes` (iya) atau `no` (tidak). Encoding ini dilakukan menggunakan `.map()`dengan cara mengganti nilai `yes` dengan `1`, dan nilai `no` dengan `0`.

### One Hot Encoding

One Hot Encoding dilakukan terhadap 2 variabel, yaitu

* `MTRANS` (Jenis transportasi yang digunakan)
* `Gender` (Jenis kelamin responden)

karena nilai-nilai pada kedua variabel tersebut tidak memiliki urutan tertentu. Encoding ini dilakukan menggunakan `.get_dummies()` dengan cara membuat kolom-kolom baru berdasarkan masing-masing nilai dari variabel tersebut, mengisinya dengan `True` atau `False` sesuai dengan keadaan, lalu menghapus kedua kolom aslinya.

### Encoding Ordinal

Encoding Ordinal dilakukan terhadap 2 variabel, yaitu
* CAEC (Konsumsi makanan di antara makan berat)
* CALC (Frekuensi konsumsi alkohol)

karena nilai-nilai pada kedua variabel tersebut memiliki urutan dengan urutannya adalah `['no', 'Sometimes', 'Frequently', 'Always']`. Encoding ini dilakukan menggunakan `OrdinalEncoder` dari library sklearn dengan cara mengurutkan nilai-nilainya, lalu diganti dengan angka sesuai urutannya.

### Data Training dan Testing

Tahapan ini dilakukan untuk membagi data menjadi 2, yaitu data training dan testing. Data training digunakan untuk melatih model dengan data yang ada, sedangkan data testing digunakan untuk menguji model yang dibuat menggunakan data yang belum dilatih. Pembagian data ini dilakukan dengan perbandingan 80% : 20% untuk data training dan data testing menggunakan `train_test_split` dari library sklearn.

## Modeling

Ada 4 algoritma *Machine Learning* yang digunakan untuk membuat model, yaitu sebagai berikut.
### 1. ***Extreme Gradient Boosting* (XGBoost)**.

Algoritma ini bekerja dengan mengimplementasikan *decision tree* yang kemudian ditingkatkan dengan gradien untuk meningkatkan kecepatan dan kinerja. Kelebihan dari algoritma ini adalah memiliki performa dan efisiensi tinggi, efektif untuk dataset bberukuran besar, dan memiliki parameter regularisasi yang mampu mencegah *overfitting*. Sementara itu, kekurangan dari algoritma ini adalah membutuhkan ketelitian terhadap hyperparameter tuning untuk mencegah *overfitting* dan komputasi yang mahal serta membutuhkan memori yang besar untuk dataset berukuran besar. <br>

Pada pemodelan ini, XGBoost diimplementasikan menggunakan `XGBClassifier` dari library `xgboost` dengan memasukkan `X_train` dan `y_train` untuk melatih model, lalu menggunakan `X_test` dan `y_test` untuk menguji model dengan data testing yang tidak ada di data training. Parameter yang digunakan pada model ini adalah `max_depth` yaitu kedalaman maksimum setiap tree, `n_estimators` yaitu jumlah tree yang akan dibuat, `random_state` yaitu mengontrol seed acak yang diberikan pada setiap iterasi, `learning rate` yaitu mengatur langkah setiap iterasi ketika meminimumkan *loss function*, dan `n_jobs` yaitu mengatur jumlah CPU threads untuk menjalankan XGBoost. Pada proyek ini, parameter yang digunakan adalah `max_depth = 10`, `n_estimators = 125`, `random_state = 30`, `learning_rate = 0.01`, `n_jobs = 20`.

### 2. ***Support Vector Machine* (SVM)**

Algoritma ini bekerja dengan mencari hyperplane terbaik untuk memisahkan kelas-kelas fitur serta menggunakan fungsi kernel untuk mentransformasikan data ke dimensi yang lebih tinggi agar dapat dipisahkan apabila pemisahan linier tidak memungkinkan. Kelebihan dari algoritma ini adalah efektif untuk dimensi tinggi, penggunaan memori yang efisien, dan dapat menggunakan fungsi kernel apapun. Sedangkan kekurangan dari algoritma ini adalah tidak cocok untuk dataset berukuran besar dan memiliki performa buruk untuk data yang noisy ataupun tidak bersih. <br>

Pada pemodelan ini, SVM diimplementasikan menggunakan `SVC` dari library `sklearn.svm` dengan memasukkan `X_train` dan `y_train` untuk melatih model, lalu menggunakan `X_test` dan `y_test` untuk menguji model dengan data testing yang tidak ada di data training. Parameter yang digunakan pada model ini adalah `kernel` yaitu tipe kernel yang digunakan untuk mentransformasikan input data, `gamma` yaitu pengaruh dari sebuah contoh training, dan `random_state` yaitu mengontrol seed acak yang diberikan pada setiap iterasi. Pada proyek ini, parameter yang digunakan adalah `kernel = 'rbf'`, `gamma = 'auto'`, `random_state = 50`.

### 3. ***K-Nearest Neighbors* (KNN)**

Algoritma ini bekerja dengan mengklasifikasikan titik data berdasarkan kelas mayoritas dari sejumlah k tetangga terdekatnya. Kelebihan dari algoritma ini adalah mudah dan simple untuk digunakan, tidak ada fase *lazy learning* sehingga cepat, dan efektif untuk dataset berukuran kecil serta untuk masalah multi-class. Sementara itu, kekurangan dari algoritma ini adalah sensitif terhadap pemilihan k dan metrik jarak serta memiliki performa buruk untuk data berdimensi tinggi (*curse of dimensionality*). <br>

Pada pemodelan ini, KNN diimplementasikan menggunakan `KNeighborsClassifier` dari library `sklearn.neighbors` dengan memasukkan `X_train` dan `y_train` untuk melatih model, lalu menggunakan `X_test` dan `y_test` untuk menguji model dengan data testing yang tidak ada di data training. Parameter yang digunakan pada model ini adalah `n_neighbors` yaitu jumlah k tetangga. Pada proyek ini, parameter yang digunakan adalah `n_neighbors = 7` karena ingin mengklasifikasi tingkat berat badan menjadi 7 sesuai dengan yang ada di data.

### 4. ***Random Forest***

Algoritma ini bekerja dengan membentuk decision trees, lalu menggunakan sampiing dengan penggantian (*bootstrapping*) dan pemilihan fitur acak untuk setiap pohon agar pohon-pohon menjadi beragam. Kelebihan dari algoritma ini adalah memiliki akurasi tinggi karena menggunakan pendekatan ensemble, mencegah *overfitting* dengan jumlah pohon yang banyak, dan mampu menangani dataset berukuran besar dan multi dimensi. Sedangkan kekurangan dari algoritma ini adalah komputasi yang besar untuk jumlah pohon yang besar dan membutuhkan memori yang besar untuk menyimpan seluruh pohon. <br>

Pada pemodelan ini, *Random Forest* diimplementasikan menggunakan `RandomForestClassifier` dari library `sklearn.ensemble` dengan memasukkan `X_train` dan `y_train` untuk melatih model, lalu menggunakan `X_test` dan `y_test` untuk menguji model dengan data testing yang tidak ada di data training. Parameter yang digunakan pada model ini adalah `n_estimators` yaitu jumlah tree yang akan dibuat, `criterion` yaitu fungsi untuk menentukan kualitas *splitting data*, `max_depth` yaitu kedalaman maksimum setiap tree, dan `random_state` yaitu mengontrol seed acak yang diberikan pada setiap iterasi. Pada proyek ini, parameter yang digunakan adalah `n_estimators = 200`, `criterion = "entropy"`, `max_depth = 10`, `random_state = 50`.

### 5. Pemilihan Model

Setelah semua model dijalankan, penulis memilih algoritma *Random Forest* sebagai model terbaik yang akan digunakan sebagai solusi untuk memprediksi obesitas karena model ini memiliki akurasi dan skor f1 tertinggi dibandingkan model lainnya, serta kesalahan klasifikasi pada matriks confusion yang lebih kecil dibanding model lainnya. Penjelasan lebih lengkap mengenai alasan ini ada di bagian selanjutnya, yaitu **evaluation**.

## Evaluation

Pada proyek ini, penilaian model menggunakan confusion matrix, akurasi, dan f1 score sebagai metrik evaluasi untuk masing-masing model. Akan dijelaskan terlebih dahulu bagaimana cara mendapatkan akurasi dan f1 score serta bagaimana cara menggunakan confusion matrix.

### Sekilas Tentang Matriks Confusion, Akurasi, dan Skor f1

Matriks Confusion merupakan sebuah tabel untuk mengukur akurasi dari model klasifikasi. Contoh dari Matriks Confusion beserta labelnya dapat dilihat pada gambar di bawah ini. 

<img src = "gambar/Confusion_Matrix_5.png"/> <br>

Setiap baris pada matriks confusion merepresentasikan nilai sesungguhnya, sedangkan setiap kolom pada matriks confusion merepresentasikan nilai yang diprediksi. Terdapat 4 label pada matriks confusion seperti yang terlihat di gambar, yaitu TP, TN, FP, dan FN.
1. *True Positive* (TP) merupakan jumlah data pada positif yang ditebak dengan benar.
2. *True Negative* (TN) merupakan jumlah data pada negatif yang ditebak dengan benar.
3. *False Positive* (FP) merupakan jumlah data yang ditebak dengan salah karena diprediksi positif, sedangkan aslinya adalah negatif. Kesalahan ini sering disebut Error Tipe 1.
4. *False Negative* (FN) merupakan jumlah data yang ditebak dengan salah karena diprediksi negatif, sedangkan aslinya adalah positif. Kesalahan ini sering disebut Error Tipe 2.

Selanjutnya, metrik evaluasi yang digunakan berdasarkan label-label yang diketahui dari matriks confusion ada 4, yaitu sebagai berikut.
1. Akurasi (*Accuracy*) merupakan proporsi data yang berhasil diprediksi dengan benar dari seluruh data yang diprediksi. Akurasi dirumuskan sebagai <br>
<img src = "gambar/Rumus_Akurasi.png"/> <br>

2. *Precision* merupakan proporsi data positif yang berhasil diprediksi dengan benar dari seluruh data yang diprediksi positif. *Precision* dirumuskan sebagai <br>
<img src = "gambar/Rumus_Precision.png"/> <br>

3. *Recall* merupakan proporsi data positif yang berhasil diprediksi dengan benar dari seluruh data yang aslinya positif. *Recall* dirumuskan sebagai <br>
<img src = "gambar/Rumus_Recall.png"/> <br>

4. Skor F1 (F1 *score*) merupakan rata-rata harmonik dari *precision* dan *recall* untuk mendapatkan sebuah metrik yang seimbang. Skor F1 dirumuskan sebagai <br>
<img src = "gambar/Rumus_SkorF1.png"/> <br>

### Penerapan Matriks Confusion, Akurasi, dan Skor f1

#### 1. Model XGBoost

Berikut merupakan matriks confusion, akurasi, dan skor f1 dari model XGBoost

<img src = "gambar/Confusion_Matrix_1.png"/> <br>

Dari gambar di atas, terdapat 11 data yang diprediksi salah pada obesitas tingkat 2 dan 7 data yang diprediksi salah pada obesitas tingkat 3. Diperoleh skor F1 nya adalah 0.93 dengan akurasi tepatnya adalah 0.933014 atau ≈93.30%.

#### 2. Model SVM

Berikut merupakan matriks confusion, akurasi, dan skor f1 dari model SVM

<img src = "gambar/Confusion_Matrix_2.png"/> <br>

Dari gambar di atas, terdapat 19 data yang diprediksi salah pada obesitas tingkat 2 dan 14 data yang diprediksi salah pada obesitas tingkat 3. Diperoleh skor F1 nya adalah 0.85 dengan akurasi tepatnya adalah 0.849282 atau ≈84.93%.

#### 3. Model KNN

Berikut merupakan matriks confusion, akurasi, dan skor f1 dari model KNN

<img src = "gambar/Confusion_Matrix_3.png"/> <br>

Dari gambar di atas, terdapat 12 data yang diprediksi salah pada obesitas tingkat 2 dan 14 data yang diprediksi salah pada obesitas tingkat 3. Diperoleh skor F1 nya adalah 0.85 dengan akurasi tepatnya adalah 0.846889 atau ≈84.69%.

#### 4. Model *Random Forest*

Berikut merupakan matriks confusion, akurasi, dan skor f1 dari model *Random Forest*

<img src = "gambar/Confusion_Matrix_4.png"/> <br>

Dari gambar di atas, terdapat 11 data yang diprediksi salah pada obesitas tingkat 2 dan 6 data yang diprediksi salah pada obesitas tingkat 3. Diperoleh skor F1 nya adalah 0.94 dengan akurasi tepatnya adalah 0.942584 atau ≈94.26%.

#### Hasil Evaluasi
Dari seluruh akurasi yang diketahui dari keempat model, dibentuk bar plot untuk melihat perbandingan nilai akurasi model sebagai berikut. 

<img src = "gambar/Barplot_4.png"/> <br>

Berdasarkan gambar di atas dan evaluasi masing-masing model untuk mengetahui skor akurasi, skor F1, dan jumlah kesalahan klasifikasi pada masing-masing model, didapat model *Random Forest* merupakan model terbaik karena memiliki skor akurasi dan skor F1 tertinggi, serta jumlah kesalahan klasifikasi yang paling sedikit, terutama pada obesitas. 

## Kesimpulan
1. Berdasarkan data yang diperoleh, sekitar 73.7% dari seluruh responden mengalami *overweight* atau obesitas, dengan 46.5% di antaranya mengalami obesitas.
2. Jenis kelamin yang berbeda memiliki tingkat obesitas yang berbeda. Dari data yang diperoleh, mayoritas laki-laki mengalami *overweight* tingkat 2, obesitas tingkat 1, dan obesitas tingkat 2. Sedangkan mayoritas perempuan mengalami obesitas tingkat 3.
3. Faktor-faktor yang memengaruhi berat badan seseorang adalah tinggi badan, usia, frekuensi konsumsi sayur-sayuran dalam sehari, dan frekuensi konsumsi air dalam sehari.
4. Seluruh penyandang obesitas tingkat 3 memiliki kesamaan dalam beberapa faktor, yaitu
   * Menggunakan transportasi umum sebagai sarana transportasi
   * Terkadang mengonsumsi alkohol
   * Melakukan aktivitas fisik kurang dari 4 hari dalam seminggu.
   * Makan berat sebanyak 3 kali dalam sehari
   * Berusia di bawah 30 tahun. 
5. Setelah menguji data menggunakan 4 model *machine learning*, yaitu ***Extreme Gradient Boosting* (XGBoost)**, ***Support Vector Machine* (SVM)**, ***K-Nearest Neighbors* (KNN)**, dan ***Random Forest*** untuk mendeteksi obesitas, diperoleh model *Random Forest* merupakan model terbaik dibandingkan model lainnya berdasarkan skor akurasi, skor F1, dan jumlah kesalahan klasifikasi yang paling sedikit.

## Referensi
1. WHO. (2024). Diakses pada 6 Juli 2024 dari https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight
2. Kułak, Klaudia Brygida, Izabela Magdalena Sztybór & Katarzyna Kamińska. "OBESITY - AN EPIDEMIC OF THE 21ST CENTURY – LITERATURE REVIEW", 2024, p. 1-10. Journal of Education, Health and Sport, diakses pada 7 Juli 2024.
3. Dicoding. Diakses pada 6 Juli 2024 dari https://www.dicoding.com/academies/319-machine-learning-terapan
4. Tim Promkes RSST - RSUP dr. Soeradji Tirtonegoro Klaten. (2022). Diakses pada 6 Juli 2024 dari https://yankes.kemkes.go.id/view_artikel/429/obesitas
5. Tandiono, Steven Marcelino & Samuel Ady SanjayaWise. "Machine Learning Approach of ObesityLevel Classification: A Systematic Literature Reviewof Methods and Factors", vol. 8, no. 1, 2023, p. 196-208. G-Tech : Jurnal Teknologi Terapan, diakses pada 7 Juli 2024.
