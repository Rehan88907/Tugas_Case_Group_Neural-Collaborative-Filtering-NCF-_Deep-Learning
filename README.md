# Tugas_Case_Group_Neural-Collaborative-Filtering-NCF-_Deep-Learning

1. Preprocessing Data

Dataset yang digunakan berupa implicit feedback (user–item interaction).
Contoh: MovieLens u.data → user_id, item_id, rating.

Langkah preprocessing:

(a) Load data:
Membaca data interaksi dari CSV.

(b) Mengonversi user_id dan item_id ke index numerik:
Embedding bekerja hanya dengan ID dalam bentuk indeks 0 → N.
Contoh:
user 10 → index 0
user 27 → index 1
Ini dilakukan dengan:

df["user_id"] = df["user_id"].astype("category").cat.codes
df["item_id"] = df["item_id"].astype("category").cat.codes

(c) Membentuk implicit feedback:
Karena kita menggunakan BCE (Binary Cross Entropy), rating 1–5 harus diubah menjadi 1 (positif).
Semua interaksi dianggap positif → label = 1.

(d) Train-test split:
Dataset dibagi menjadi:
train set 80%
test set 20%

(e) Membuat struktur user → item yang pernah ditonton

2. Pembentukan Embedding (Representasi User dan Item)
NCF tidak menggunakan matrix factorization tradisional, tapi menggunakan neural embedding, yaitu vektor representasi user dan item.

(a) User embedding:
Jika ada 943 user dan dimensi embedding = 16:
Maka embedding matrix berukuran 943 × 16
Setiap user punya vektor 16 dimensi

(b) Item embedding:
Jika ada 1682 item dan dimensi embedding = 16:
Embedding = 1682 × 16
Embedding ini dilatih bersama neural network menggunakan backpropagation.

Tujuan embedding
Merepresentasikan preferensi user dan karakteristik item dalam bentuk vektor yang bisa dipelajari.

3. Arsitektur Model NCF (NeuMF)

NCF menggabungkan dua pendekatan:

(a) GMF — Generalized Matrix Factorization:

Melakukan operasi element-wise product antara embedding user dan item:
𝑝𝑢 ⊙ 𝑞𝑖
	​GMF menangkap interaksi linear.

(b) MLP — Multi Layer Perceptron:

Menggabungkan embedding user dan item dengan cara concatenate, lalu melewati beberapa dense layer berhingga nonlinear:

𝑀𝐿𝑃([𝑝𝑢,𝑞𝑖])
MLP menangkap interaksi nonlinear dan kompleks.
	​
(c) NeuMF = GMF + MLP:
Kedua bagian digabungkan:

𝑜𝑢𝑡𝑝𝑢𝑡=𝑠𝑖𝑔𝑚𝑜𝑖𝑑([𝐺𝑀𝐹,𝑀𝐿𝑃]⋅ℎ)

Output adalah probabilitas user menyukai item.

4. Negative Sampling (Wajib untuk Implicit Feedback):
Karena dataset implicit hanya berisi interaksi positif, kita membutuhkan contoh negatif (0).
Negative sampling dilakukan di train.py:

-Untuk tiap interaksi positif (user, item_pos)

-Pilih beberapa item yang tidak pernah diinteraksi user

-Labeli dengan 0

Misalnya:
(user 10, item 15) → POSITIF = 1
(user 10, item 300) → NEGATIF = 0
(user 10, item 112) → NEGATIF = 0
Inilah sebabnya BCE berhasil digunakan.

5. Training (Proses Pembelajaran)
Model dilatih menggunakan:

-Binary Cross Entropy Loss

-Optimizer ADAM

-Mini-batch training menggunakan DataLoader

Tujuan:
Prediksi(user,item)→1 jika user suka item


Selama training:

-Positif mendorong output → 1

-Negatif mendorong output → 0

Loss turun seperti yang kamu lihat: Loss: 22 → 0.159 → 0.054

6. Evaluasi (Top-K Recommendation Metrics):

Model rekomendasi tidak diukur dengan akurasi biasa.

Digunakan ranking metrics: 

(a) Hit Ratio @ K:
Mengukur apakah item yang benar (ground truth) muncul dalam daftar rekomendasi top-K.

𝐻𝑅@10=1jika item test masuk top 10

(b) NDCG @ K:
Mengukur apakah item yang benar berada di posisi atas (ranking-aware).

Jika item muncul di:

-posisi #1 → skor besar
-posisi #10 → skor kecil

Hasil kamu: 
Hit Ratio@10: 0.1195
NDCG@10     : 0.0531
Ini NORMAL untuk NCF sederhana 3 epoch.

7. Kesimpulan Alur Lengkap
Berikut alur NCF dari awal sampai akhir:

A.Load dataset

B.Encode user_id & item_id

C.Konversi ke implicit feedback (label=1)

D.Buat struktur user_items

E.Train-test split

F.Negative sampling

G.User embedding + Item embedding

H.GMF block

I.MLP block

J.Gabungkan GMF + MLP → NeuMF

K.Training dengan BCE

L.Evaluasi ranking: HR@10, NDCG@10

Ini digunakan untuk negative sampling dan evaluasi top-k.
