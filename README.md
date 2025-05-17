# Laporan Proyek Machine Learning - Siti Septiyah Agustin

## Project Overview

## Latar Belakang

Indonesia memiliki banyak tempat wisata menarik yang tersebar di berbagai wilayah, namun sering kali wisatawan kesulitan memilih destinasi yang sesuai dengan preferensi pribadi. Dalam era digital, sistem rekomendasi dapat membantu pengguna menemukan destinasi wisata yang paling relevan dengan preferensi mereka, seperti jenis wisata, rating, dan popularitas.

Masalah ini perlu diselesaikan karena pemilihan destinasi yang tidak sesuai dapat mengurangi kepuasan pengalaman wisatawan, yang pada akhirnya berdampak pada industri pariwisata secara keseluruhan. Selain itu, banyaknya pilihan membuat wisatawan mengalami information overload, sehingga dibutuhkan sistem yang dapat melakukan penyaringan otomatis.

Masalah tersebut diselesaikan dengan membangun sistem rekomendasi berbasis machine learning yang menganalisis fitur-fitur tempat wisata seperti deskripsi, kategori, dan kata kunci. Dengan pendekatan content-based filtering dan transformasi teks menggunakan TF-IDF, sistem dapat menghitung kemiripan antar destinasi dan menyarankan pilihan yang relevan kepada pengguna.

Menurut Shrestha et al. [[1] https://doi.org/10.3390/computation12030059], pendekatan berbasis data dan machine learning dapat meningkatkan akurasi rekomendasi destinasi wisata dengan mempertimbangkan preferensi pengguna dan konteks perjalanan. Selain itu, Herzog dan Wörndl [[2] https://doi.org/10.1145/2645710.2645740] menyatakan bahwa kombinasi pendekatan content-based dan collaborative filtering mampu meningkatkan relevansi hasil rekomendasi.

## Business Understanding


### Problem Statements

1. **Bagaimana memberikan rekomendasi tempat wisata yang sesuai dengan preferensi pengguna berdasarkan data yang tersedia?**
   Wisatawan sering mengalami kesulitan dalam menemukan destinasi wisata yang sesuai dengan preferensi atau minat pribadi mereka di tengah banyaknya pilihan yang tersedia di Indonesia.

2. **Bagaimana meningkatkan pengalaman pengguna dalam memilih tempat wisata melalui sistem otomatis?**
   Platform pariwisata atau agen perjalanan seringkali tidak menyediakan sistem rekomendasi yang dipersonalisasi untuk pengguna, sehingga mengurangi kepuasan dan efisiensi dalam merencanakan perjalanan.

### Goals

1. **Mengembangkan sistem rekomendasi yang dapat menyarankan destinasi wisata dengan pendekatan machine learning.**
   Membangun sistem rekomendasi yang mampu memberikan saran destinasi wisata yang relevan dan sesuai dengan preferensi pengguna, menggunakan teknik pemrosesan bahasa alami dan pembelajaran mesin.

2. **Menyediakan antarmuka rekomendasi yang mudah digunakan dan dapat menampilkan hasil top-N tempat wisata yang paling relevan.**
   Menyediakan alat bantu berbasis data yang dapat diintegrasikan ke dalam platform wisata digital untuk meningkatkan pengalaman pengguna dan mendorong eksplorasi destinasi yang lebih luas.


    ### Solution statements
    1. **Content-based Filtering**
       Menggunakan algoritma TF-IDF (Term Frequency-Inverse Document Frequency) untuk mengekstraksi fitur penting dari deskripsi destinasi wisata, lalu menghitung kesamaan antar destinasi menggunakan cosine similarity. Dengan ini, sistem dapat merekomendasikan destinasi serupa berdasarkan deskripsi yang sudah disukai pengguna.

    2. **Pendekatan Visualisasi dan Eksplorasi Data**
       Dengan menerapkan visualisasi grafik distribusi, sistem mampu memberikan wawasan tambahan terhadap karakteristik destinasi yang populer atau saling berkaitan.

    3. **Potensi Pengembangan (Hybrid Recommender System)**
       Meskipun proyek saat ini menggunakan pendekatan berbasis konten, sistem ini dapat dikembangkan lebih lanjut menjadi model hybrid, yang menggabungkan teknik collaborative filtering untuk menghasilkan rekomendasi yang lebih akurat dengan mempertimbangkan interaksi pengguna.

## Data Understanding

Link Dataset: https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination

Proyek ini menggunakan dataset Indonesia Tourism Destination yang dapat diakses melalui [Kaggle](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination). Dataset ini adalah dataset yang berisi beberapa tempat wisata di 5 kota besar di Indonesia, yaitu Jakarta, Yogyakarta, Semarang, Bandung, dan Surabaya. Data yang digunakan dalam proyek ini berasal dari 4 file, dan yang akan saya gunakan yakni `tourism_with_id.csv`
File `tourism_ with _id.csv` berisi informasi mengenai tempat wisata di 5 kota besar di Indonesia dengan total sekitar 400 tempat.  

### Deskripsi Atribut Dataset

Berikut ini adalah deskripsi lengkap dari setiap atribut dalam dataset:

**1. File `tourism_with_id.csv`**

| No. | Kode Atribut | Deskripsi                                                                                     | Tipe Data |
|-----|--------------|-----------------------------------------------------------------------------------------------|-----------|
| 1   | Place_Id     | ID unik untuk setiap tempat wisata (misalnya 1 untuk Monumen Nasional, 2 untuk Kota Tua, dll).| Integer   |
| 2   | Place_Name   | Nama dari tempat wisata (misalnya Monumen Nasional, Kota Tua, Dunia Fantasi, dll).            | String    |
| 3   | Description  | Deskripsi singkat tentang tempat wisata.                                                      | String    |
| 4   | Category     | Kategori tempat wisata (misalnya Budaya, Taman Hiburan, Alam, dll).                           | String    |
| 5   | City         | Kota tempat tempat wisata berada (misalnya Jakarta).                                          | String    |
| 6   | Price        | Harga tiket atau biaya untuk mengunjungi tempat wisata dalam satuan IDR (Rupiah).             | Integer   |
| 7   | Rating       | Rating tempat wisata berdasarkan penilaian pengguna (misalnya 4.6 untuk Monumen Nasional).    | Float     |
| 8   | Time_Minutes | Waktu yang diperlukan untuk mengunjungi tempat wisata dalam satuan menit.                     | Float     |
| 9   | Coordinate   | Koordinat geografis tempat wisata dalam format latitude dan longitude.                        | String    |
| 10  | Lat          | Nilai latitude tempat wisata, menunjukkan posisi vertikal di peta bumi.                       | Float     |
| 11  | Long         | Nilai longitude tempat wisata, menunjukkan posisi horizontal di peta bumi.                    | Float     |
| 12  | Unnamed: 11  | Kolom tidak terpakai (semua nilai NaN).                                                       | Float     |
| 13  | Unnamed: 12  | Kolom menunjukkan urutan atau ID pengurutan tempat wisata dalam dataset                       | Integer   |


### Exploratory Data Analysis - EDA

Exploratory Data Analysis (EDA) merupakan langkah awal dalam proses analisis data yang bertujuan untuk mengeksplorasi struktur dataset, mengenali pola-pola yang muncul, serta memahami hubungan antar variabel. Tahapan ini berguna untuk mendeteksi anomali, menilai distribusi data, dan menggali insight awal yang akan menunjang proses persiapan data dan pemodelan. Pada proyek ini, EDA dilakukan untuk menganalisis karakteristik destinasi wisata, sebaran rating, serta keterkaitan antara pengguna dan destinasi yang mereka kunjungi. Proses EDA mencakup pemeriksaan struktur data, visualisasi distribusi fitur, serta analisis korelasi antar atribut yang dianggap relevan.

**Informasi Dataset `tourism_with_id.csv`**

**Struktur Data dengan `data_tourism_with_id.info()`**  

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 437 entries, 0 to 436
Data columns (total 13 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Place_Id      437 non-null    int64  
 1   Place_Name    437 non-null    object 
 2   Description   437 non-null    object 
 3   Category      437 non-null    object 
 4   City          437 non-null    object 
 5   Price         437 non-null    int64  
 6   Rating        437 non-null    float64
 7   Time_Minutes  205 non-null    float64
 8   Coordinate    437 non-null    object 
 9   Lat           437 non-null    float64
 10  Long          437 non-null    float64
 11  Unnamed: 11   0 non-null      float64
 12  Unnamed: 12   437 non-null    int64  
dtypes: float64(5), int64(3), object(5)
memory usage: 44.5+ KB
```
**Dari tampilan di atas terlihat bahwa:**

- Dataset ini terdiri dari **437 entri** (baris) dan **13 kolom**.

- Terdapat tiga tipe data utama dalam dataset:
  - **Integer** (`int64`): `Place_Id`, `Price`, dan `Unnamed: 12`.
  - **Float** (`float64`): `Rating`, `Time_Minutes`, `Lat`, `Long`, dan `Unnamed: 11`.
  - **String** (`object`): `Place_Name`, `Description`, `Category`, `City`, dan `Coordinate`.

- Nilai null:
  - Kolom `Time_Minutes` memiliki **205 nilai non-null** dari total 437 entri, yang berarti terdapat **232 nilai yang hilang**.
  - Kolom `Unnamed: 11` tidak memiliki nilai yang terisi (**0 non-null**), sehingga kolom ini sepenuhnya kosong.
  - Kolom lainnya memiliki **437 nilai non-null**, artinya tidak ada nilai yang hilang kecuali pada `Time_Minutes` dan `Unnamed: 11`.
  - terdapat missing value pada fitur **Time_Minutes** dan **Unnamed: 11**, namun fitur tersebut tidak digunakan untuk sistem rekomendasi ini, jadi tidak perlu penanganan missing value
- Dataset ini menggunakan **44.5 KB** memori.

**Distribusi Rating Destinasi Wisata**

Dengan menggunakan `sns.histplot()` dan `sns.boxplot`, didapatkan visualisasi distribusi rating destinasi wisata sebagai berikut.

![image](https://github.com/user-attachments/assets/fee8d693-45ba-439c-80e6-1f1a2337a052)


**Insight:** 
Rating yang diberikan oleh pengguna pada tempat wisata umumnya cukup tinggi, dengan mayoritas rating berada dalam rentang 4.3 hingga 4.6. Visualisasi menggunakan boxplot menunjukkan distribusi yang relatif seragam tanpa perbedaan mencolok antar kuartil, menandakan konsistensi dalam penilaian pengguna. Secara keseluruhan, data ini mengindikasikan bahwa tempat wisata dihargai dengan baik oleh pengunjung, dengan hanya sedikit tempat yang memiliki rating sangat rendah atau sangat tinggi.

**Distribusi Harga Tempat Wisata**

Dengan menggunakan `sns.histplot()` dan `sns.boxplot`, didapatkan visualisasi distribusi rating destinasi wisata sebagai berikut.

![image](https://github.com/user-attachments/assets/c5400fb8-0e5b-4e0a-8dfe-d68993457805)


**Insight:** 
Distribusi harga menunjukkan bahwa sebagian besar tempat wisata memiliki harga yang sangat rendah atau bahkan gratis, dengan hanya beberapa tempat yang memiliki harga sangat tinggi dan tercatat sebagai outlier. Hal ini mencerminkan bahwa mayoritas destinasi wisata bersifat terjangkau dan lebih ditujukan untuk kalangan umum, sementara hanya sedikit destinasi yang menargetkan segmen pasar premium. Secara keseluruhan, harga tiket tempat wisata cenderung terpusat pada kisaran rendah, dengan harga tinggi muncul sebagai kasus khusus yang tidak umum.

**Distribusi Kategori Wisata**

Dengan menggunakan `sns.countlot()` dan `sns.boxplot`, didapatkan grafik jumlah, rating, dan harga destinasi wisata per kategori sebagai berikut.

![image](https://github.com/user-attachments/assets/b1f3ac14-11a8-4fdd-a021-cbe65c2a1565)


**Insight:** 
Kategori Taman Hiburan dan Budaya mendominasi dalam hal jumlah tempat wisata, dengan sebagian besar memiliki rating tinggi serta harga tiket yang rendah hingga sedang. Sementara itu, kategori Cagar Alam mencakup beberapa tempat dengan harga yang cukup tinggi. Kategori Pusat Perbelanjaan dan Tempat Ibadah cenderung terdiri dari jumlah tempat wisata yang lebih sedikit dan berfokus pada harga yang lebih rendah. Secara keseluruhan, semua kategori umumnya memiliki rating yang tinggi, meskipun terdapat beberapa outlier dalam penilaiannya.

**Distribusi Kota dengan tujuan terbanyak**

![image](https://github.com/user-attachments/assets/c49b736b-befe-4cb8-b8f8-8aa8c3d14af4)


**Insight:** 
Grafik ini menunjukkan bahwa Bandung memiliki jumlah destinasi pariwisata terbanyak, diikuti Jakarta dan Yogyakarta, masing-masing dengan 80-90 destinasi. Semarang dan Surabaya memiliki jumlah destinasi lebih rendah, dengan Surabaya paling sedikit. Secara keseluruhan, kota-kota besar seperti Bandung, Jakarta, dan Yogyakarta mendominasi sektor pariwisata Indonesia.

## Data Preparation

Pada tahap **data preparation**, dilakukan serangkaian langkah untuk memastikan bahwa data yang digunakan dalam proses pemodelan bersih, terstruktur, dan siap untuk dianalisis. Tahapan ini mencakup pembersihan data, penggabungan dataset, pengolahan teks, encoding variabel kategorikal, normalisasi rating, dan pembagian data menjadi set pelatihan dan pengujian. Berikut adalah uraian mendetail mengenai setiap langkah yang dilakukan:

### 1. Menangani missing value

tahap ini akan saya lewati, untuk mempercepat preparation. 

- **alasan**: dapat dilihat pada tahap EDA dataset **data_tourism_with_id** meskipun terdapat missing value pada fitur **Time_Minutes** dan **Unnamed: 11**, saya tidak melakukan penanganan missing value, karena fitur tersebut tidak akan saya gunakan (dihapus/didrop) untuk sistem rekomendasi ini, jadi tidak perlu penanganan missing value.

### 2. Menghapus Kolom yang Tidak Perlu

- Kolom `Unnamed: 11` dan `Unnamed: 12`dihapus karena `Unnamed: 11` sepenuhnya kosong dan `Unnamed: 12` juga tidak relevan.
- Kolom `Time_Minutes`, `Coordinate`, `Lat`, dan `Long` dihapus karena sistem rekomendasi akan dibangun berdasarkan kategori dan deskripsi, bukan informasi dari kolom-kolom tersebut.

```python
columns_to_drop = ['Unnamed: 11', 'Unnamed: 12', 'Coordinate', 'Lat', 'Long', 'Time_Minutes']
tourism_with_id = tourism_with_id.drop(columns=[col for col in columns_to_drop if col in tourism_with_id.columns], errors='ignore')
```
- **Alasan**: Mengurangi dimensi data yang tidak relevan membantu mempercepat proses analisis dan meminimalisir kompleksitas model, serta mengurangi kemungkinan overfitting.

### 3. Cleaning data

- membersihkan teks dengan menghapus karakter-karakter khusus, angka, dan spasi ekstra, serta mengubah teks menjadi huruf kecil
- Membersihkan Kolom Description

```python
def clean_text(text):
    """
    Membersihkan teks dengan menghapus karakter khusus, angka, dan spasi ekstra.
    """
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)  # Hapus karakter non-alfanumerik
    text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi ekstra
    return text.lower()

data_tourism_with_id_clean['Description'] = data_tourism_with_id_clean['Description'].apply(clean_text)
```
- **Alasan**: Pembersihan teks meningkatkan kualitas data dan memastikan bahwa proses analisis atau pemodelan berfokus pada kata-kata yang relevan, mengurangi gangguan dari karakter yang tidak diinginkan.

### 4. Penggabungan Teks

- Menggabungkan Kolom Teks yang Relevan
- Membersihkan Kolom Combined_Text

```python
data_tourism_with_id_clean['Combined_Text'] = (
    data_tourism_with_id_clean['Place_Name'] + ' ' +
    data_tourism_with_id_clean['Description'] + ' ' +
    data_tourism_with_id_clean['Category'] + ' ' +
    data_tourism_with_id_clean['City']
)

data_tourism_with_id_clean['Combined_Text'] = data_tourism_with_id_clean['Combined_Text'].apply(clean_text)
```
- **Alasan**: Menggabungkan informasi dari beberapa kolom terkait ke dalam satu kolom teks membuatnya lebih mudah untuk diterapkan pada teknik pemrosesan teks, seperti TF-IDF, dan memungkinkan model untuk menganalisis seluruh konteks tempat wisata.

### 5. Normalisasi Price Tag

- Price_Normalized berisi harga yang telah dinormalisasi dalam rentang 0 hingga 1.

Price  Price_Normalized
0   20000          0.022222
1       0          0.000000
2  270000          0.300000
3   10000          0.011111
4   94000          0.104444

```python
data_tourism_with_id_clean['Price_Normalized'] = scaler.fit_transform(data_tourism_with_id_clean[['Price']])
```

- **Alasan**: Normalisasi harga membantu agar variabel harga dapat digunakan secara efektif dalam model dan mencegah algoritma machine learning memberi bobot yang berlebihan pada variabel yang memiliki rentang nilai yang lebih besar.

### 6. mengubah Combined_Text menjadi vektor TF-IDF

- mengubah kolom Combined_Text menjadi representasi numerik menggunakan TF-IDF, di mana kata-kata yang lebih sering muncul dalam dokumen tetapi jarang muncul di seluruh dataset mendapatkan bobot lebih tinggi.
- Matriks tfidf_matrix yang dihasilkan dapat digunakan untuk perhitungan cosine similarity atau tugas pembelajaran mesin lainnya yang membutuhkan representasi vektor dari teks.

```python
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(data_tourism_with_id_clean['Combined_Text'])
columns = tfidf.get_feature_names_out()
tfidf_matrix.shape
```
visualisasi mengubah Combined_Text menjadi vektor TF-IDF sebagai berikut.

![Normalisasi Price Tag](/img/chage.png)

- **Alasan**: Menggunakan TF-IDF memungkinkan model untuk menangkap pentingnya kata-kata dalam konteks setiap destinasi wisata, sehingga meningkatkan kemampuan sistem rekomendasi untuk menemukan kesamaan antar destinasi berdasarkan deskripsi teks.


### 7. menghitung cosine similarity antara tempat-tempat wisata menggunakan TF-IDF matrix

- mengilustrasikan tingkat kemiripan antara tempat wisata berdasarkan teks deskripsi
- menampilkan sampel acak dari cosine similarity matrix

```python
cosine_sim_df = pd.DataFrame(cosine_sim, index=data_tourism_with_id_clean['Place_Name'], columns=data_tourism_with_id_clean['Place_Name'])
print('Shape:', cosine_sim_df.shape)
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)
```
visualisasi menghitung cosine similarity antara tempat-tempat wisata menggunakan TF-IDF matrix sebagai berikut.

![Normalisasi Price Tag](/img/cons.png)

- **Alasan**: Menghitung cosine similarity memungkinkan sistem untuk mengukur seberapa mirip deskripsi tempat wisata satu dengan yang lainnya berdasarkan teks, yang menjadi dasar rekomendasi dalam model content-based filtering.

## Modeling

Pada tahap ini, dibangun dan dievaluasi dua sistem rekomendasi yang berbeda, yaitu **Content-based Filtering** dan **Collaborative Filtering**. Kedua pendekatan ini menggunakan metode dan algoritma yang khas untuk memberikan rekomendasi destinasi wisata kepada pengguna.

### 1. Content-based Filtering

**Content-based Filtering** menggunakan informasi konten dari destinasi wisata, seperti deskripsi dan kategori, untuk memberikan rekomendasi yang relevan kepada pengguna. Pendekatan ini menganalisis kesamaan antara destinasi wisata berdasarkan atribut-atribut tersebut.

**Langkah-langkah:**

- Memastikan bahwa nama destinasi wisata (`wisata_nama`) yang dimasukkan tersedia dalam data similarity.
- Mengambil skor kemiripan (cosine similarity) dari destinasi wisata yang diminta terhadap seluruh destinasi lain yang ada.
- Menentukan indeks dari destinasi-destinasi yang memiliki tingkat kemiripan tertinggi menggunakan fungsi `argpartition`.
- Mengurutkan destinasi wisata berdasarkan skor kemiripan tertinggi, kemudian memilih destinasi selain yang diminta dengan menggunakan `drop`.
- Menggabungkan hasil rekomendasi dengan data asli destinasi wisata untuk mendapatkan informasi lengkap seperti nama tempat dan kategori.
- Mengembalikan sejumlah `k` rekomendasi teratas yang paling mirip dengan destinasi wisata yang diminta.

**Implementasi Kode:**

```python
def resto_recommendations(wisata_nama, similarity_data=cosine_sim_df, items=data_tourism_with_id_clean[['Place_Name', 'Category']], k=5):
    """
    Recommends tourism places based on similarity.

    Args:
        wisata_nama: The name of the tourism place to find recommendations for.
        similarity_data: The DataFrame containing the cosine similarity scores.
        items: The DataFrame containing tourism data (default: data_tourism_with_id_clean).
        k: The number of recommendations to return.

    Returns:
        A DataFrame of recommended tourism places.
    """
    index = similarity_data.loc[:,wisata_nama].to_numpy().argpartition(
        range(-1, -k, -1))
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(wisata_nama, errors='ignore')

    return pd.DataFrame(closest).merge(items).head(k)
```

**Hasil Rekomendasi:**

Sebagai contoh, kami melakukan rekomendasi untuk destinasi wisata **Monumen Nasional**.

| No. | Place Name                     | Category |
--------------------------------------------------|
| 0   | Monumen Nasional               | Budaya   |
| 1	  | Monumen Selamat Datang         | Budaya   |
| 2	  | Monumen Bandung Lautan Api     | Budaya   |
| 3	  | Monumen Bambu Runcing Surabaya | Budaya   |
| 4	  | Monumen Tugu Pahlawan	       | Budaya   |

**Kelebihan dan Kekurangan:**

| **Kelebihan**                                                                 | **Kekurangan**
|---------------------------------------------------------------------------------------------------------------------------------| 
| Rekomendasi disesuaikan dengan preferensi spesifik pengguna berdasarkan     | Sistem sangat bergantung pada kualitas dan kelengkapan  
  histori atau atribut item yang disukai.                                       atribut item (metadata). Jika data deskriptif terbatas,
                                                                                hasil rekomendasi kurang akurat. 
| Bisa memberikan rekomendasi bahkan jika hanya satu pengguna yang aktif,     | Hanya merekomendasikan item yang sangat mirip dengan yang
  karena tidak membutuhkan data dari pengguna lain.                             sudah disukai pengguna, sehingga sulit menemukan  hal
                                                                                baru atau berbeda (kurang eksplorasi).   
| Jika item baru memiliki metadata (fitur deskriptif), maka dapat segera      | Jika item baru tidak memiliki metadata, maka sistem tidak
  direkomendasikan.                                                             bisa memproses atau merekomendasikannya. 
  
| Sistem dapat dipahami dan ditelusuri, karena rekomendasi didasarkan pada    | Misalnya, pengguna menyukai variasi atau kombinasi fitur 
  kesamaan fitur item.                                                          tertentu, yang sulit diidentifikasi hanya dari fitur item.

## Evaluation

Evaluasi merupakan langkah untuk menilai sejauh mana performa sistem rekomendasi yang telah dibangun memenuhi tujuan yang telah ditetapkan. Dalam proyek ini, terdapat tiga metrik evaluasi utama yaitu:
- **Precision** untuk menilai pendekatan sistem **Content-based Filtering**
- **Root Mean Squared Error (RMSE)** serta **Mean Absolute Error (MAE)** untuk menilai pendekatan sistem rekomendasi **Collaborative Filtering**.

## Evaluasi Model Content-Based Filtering
### **Evaluasi Precision\@10 dengan Konten Deskripsi dan Kategori**

Fungsi `evaluate_precision` digunakan untuk mengevaluasi seberapa akurat sistem rekomendasi dalam menyarankan destinasi wisata yang relevan berdasarkan informasi konten, yaitu deskripsi dan kategori dari destinasi wisata.

---

### 1. **Metrik Evaluasi yang Digunakan**

**a. Precision**
Precision mengukur proporsi rekomendasi yang relevan di antara 10 rekomendasi teratas yang diberikan kepada pengguna. Metrik ini memberikan indikasi seberapa akurat rekomendasi yang dihasilkan oleh sistem.

**Formula:**
$\text{Precision} = \frac{\text{TP}}{TP+FP}$

Di mana:

 - \$TP\$ (True Positive) adalah jumlah rekomendasi yang relevan di antara 10 rekomendasi teratas,
 - \$FP\$ (False Positive) adalah jumlah rekomendasi yang tidak relevan di antara 10 rekomendasi teratas.

**Interpretasi:**

- Jika nilai precision tinggi, artinya sebagian besar rekomendasi yang diberikan relevan dengan preferensi pengguna.
- Jika nilai precision rendah, artinya sebagian besar rekomendasi yang diberikan tidak relevan.


### **Langkah-langkah Evaluasi dalam Kode**

- Algoritma memilih satu atau beberapa destinasi wisata (`places_to_evaluate`) yang akan dievaluasi kualitas rekomendasinya.

- Untuk setiap destinasi wisata yang dipilih (`place`), algoritma menggunakan fungsi `tourism_recommendations` untuk menghasilkan daftar 10 rekomendasi destinasi wisata teratas (`recommended_place_ids`) berdasarkan kesamaan konten (gabungan deskripsi dan kategori / `similarity_tags`).

- Jika tidak ditemukan rekomendasi (daftar kosong), proses dilanjutkan ke destinasi berikutnya.

- Dari daftar rekomendasi yang dihasilkan, algoritma mengambil informasi detail mengenai setiap destinasi wisata yang direkomendasikan, seperti `Place_Name`, `Category`, `Rating`, dan `Description`.

- Selanjutnya, sistem mengambil data kategori (`Category`) dan deskripsi yang telah diproses (`Description`) dari destinasi input (`place`) untuk kemudian digunakan dalam perhitungan kesamaan konten.

- Sistem menghitung nilai kemiripan deskripsi (`Description_Similarity`) antara destinasi wisata input (`place`) dan setiap destinasi yang direkomendasikan menggunakan matriks kesamaan deskripsi (`similarity_desc`). Indeks baris dari destinasi input digunakan untuk mengambil baris dari matriks kemiripan.


### **Menentukan Relevansi Rekomendasi**

Setiap rekomendasi dievaluasi apakah **relevan** atau **tidak relevan** berdasarkan dua kriteria:

1. **Kecocokan Kategori**
   Jika kategori destinasi wisata yang direkomendasikan sama dengan kategori destinasi input.

2. **Kesamaan Deskripsi**
   Jika nilai kesamaan deskripsi antara destinasi input dan destinasi yang direkomendasikan **lebih besar dari atau sama dengan threshold** (`desc_threshold`), misalnya 0.5.

Jika salah satu dari kedua kriteria tersebut terpenuhi, maka destinasi dianggap relevan (`Relevance = 1`), jika tidak maka dianggap tidak relevan (`Relevance = 0`).

### **Menghitung Precision\@10**

- **True Positives (TP):** Jumlah rekomendasi yang relevan (nilai `Relevance = 1` dari total 10 rekomendasi).
- **False Positives (FP):** Jumlah rekomendasi yang tidak relevan, yaitu `10 - TP`.

Precision dihitung dengan rumus:
$\text{Precision} = \frac{TP}{10}$

### **Output dan Rata-rata Precision**

Untuk setiap destinasi yang dievaluasi:

- Sistem mencetak rekomendasi yang diberikan dan nilai kesamaan deskripsi dari masing-masing destinasi.
- Precision\@10 untuk destinasi tersebut dicetak dalam persentase.
- Precision juga disimpan dalam list untuk kemudian dihitung rata-ratanya.

Terakhir, fungsi ini mengembalikan dua output:

1. **DataFrame `evaluation_results`** berisi nama destinasi dan Precision\@10-nya.
2. **Rata-rata nilai Precision\@10** dari seluruh destinasi yang dievaluasi.

### **Implementasi Evaluasi Precision**

Berikut implementasi penggunaannya:

```python
places_to_evaluate = ['Museum Nasional']
recommendation_data = data_tourism_with_id_clean

# Matriks kesamaan berdasarkan Deskripsi + Kategori
tfidf_tags = TfidfVectorizer()
tfidf_matrix_tags = tfidf_tags.fit_transform(recommendation_data['Combined_Text'])
similarity_tags = cosine_similarity(tfidf_matrix_tags)

# Matriks kesamaan berdasarkan Deskripsi
tfidf_desc = TfidfVectorizer()
tfidf_matrix_desc = tfidf_desc.fit_transform(recommendation_data['Description'])
similarity_desc = cosine_similarity(tfidf_matrix_desc)

# Evaluasi Precision@10
evaluation_df, avg_precision = evaluate_precision(
    recommendation_data,
    similarity_tags,
    similarity_desc,
    places_to_evaluate,
    top_n=10,
    desc_threshold=0.5
)

# Tampilkan hasil evaluasi
print("\n=== Hasil Evaluasi Precision@10 ===")
display(evaluation_df)
```
### Hasil evaluasi
top-10 
|**No.**| **Place\_Name**                      | **Category** | **Rating** | **Description\_Similarity** | **Relevance** |
|-------| ------------------------------------ | ------------ | ---------- | --------------------------- | ------------- |
|1      | Museum Wayang                        | Budaya       | 4.5        | 0.446066                    | 1             |
|2      | Museum Macan (Modern and Contemporary) Budaya       | 4.5        | 0.379759                    | 1             |
|3      | Museum Bahari Jakarta                | Budaya       | 4.4        | 0.358572                    | 1             |
|4      | Museum Seni Rupa dan Kramik          | Budaya       | 4.4        | 0.454786                    | 1             |
|5      | Museum Joang 45                      | Budaya       | 4.0        | 0.368159                    | 1             |
|6      | Museum Sumpah Pemuda                 | Budaya       | 4.7        | 0.382151                    | 1             |
|7      | Museum Tengah Kebun                  | Budaya       | 4.6        | 0.449185                    | 1             |
|8      | Museum Sonobudoyo Unit I             | Budaya       | 4.6        | 0.373011                    | 1             |
|9      | Museum Barli                         | Budaya       | 4.4        | 0.367332                    | 1             |
|10     | Museum Gedung Sate                   | Budaya       | 4.6        | 0.378406                    | 1             |

True Positive (TP): 10
False Positive (FP): 0
Precision@10 for 'Museum Nasional': 100.00%

Tabel di atas menunjukkan hasil rekomendasi yang relevan untuk Museum Nasional, dengan seluruh 10 rekomendasi dianggap relevan karena memiliki kesamaan kategori dan deskripsi yang tinggi.

Kesimpulan Evaluasi: Model Content-Based Filtering menunjukkan hasil evaluasi yang sangat baik untuk Museum Nasional, dengan Precision@10 sebesar 100%, yang berarti bahwa seluruh 10 rekomendasi yang diberikan oleh sistem relevan dan sesuai dengan deskripsi dan kategori tempat wisata. Dengan hasil ini, sistem rekomendasi yang digunakan dapat dianggap cukup efektif dalam memberikan rekomendasi yang sesuai dengan preferensi pengguna berdasarkan konten deskripsi dan kategori tempat wisata.

## **Kesimpulan secara menyeluruh**
- Model Content-Based Filtering telah menunjukkan performa yang sangat baik, dengan Precision@10 sebesar 100%, yang berarti seluruh rekomendasi yang diberikan oleh sistem sepenuhnya relevan dengan entitas yang diuji (Museum Nasional).

- Hasil evaluasi memperlihatkan bahwa model mampu mengenali tempat-tempat wisata yang memiliki kesamaan kategori dan deskripsi dengan sangat akurat, terutama dalam domain tempat wisata budaya.

- Dengan tidak adanya False Positive (FP = 0) Precision@10: 100.00%, model ini memberikan jaminan bahwa rekomendasi yang muncul memang sesuai dengan preferensi konten yang dianalisis, sehingga meminimalkan noise dalam hasil rekomendasi.

- Model ini sudah sangat layak digunakan dalam skenario nyata, terutama untuk sistem rekomendasi wisata berbasis konten, karena dapat memberikan hasil yang konsisten dan tepat sasaran.

- Ke depannya, model ini juga berpotensi untuk dikembangkan lebih lanjut sebagai bagian dari pendekatan hybrid dengan metode lain seperti Collaborative Filtering, untuk mencakup lebih banyak variasi pengguna dan preferensi yang lebih kompleks.

- Secara keseluruhan, Content-Based Filtering dalam studi ini telah berhasil memenuhi tujuan utama sistem rekomendasi, yaitu memberikan saran yang akurat dan relevan berdasarkan karakteristik tempat wisata yang dianalisis.
