<br />
<div align="center">
  <h3 align="center">Calibur Detection System</h3>

  <p align="center">
    Aplikasi Deteksi Produk Snack Cerdas Menggunakan Deep Learning (TinyViT + FCOS)
    <br />
    <a href="#">View Demo</a>
    ·
    <a href="#">Report Bug</a>
    ·
    <a href="#">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#features">Features</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
  </ol>
</details>

## About The Project

**Calibur Detection System** adalah aplikasi cerdas yang dirancang untuk kebutuhan penelitian Skripsi, bertujuan untuk mendeteksi dan mengklasifikasikan produk makanan ringan (snack) secara *real-time*. Menggunakan arsitektur *Deep Learning* modern (TinyViT + FCOS), aplikasi ini dapat mengenali berbagai varian produk dengan akurasi tinggi meskipun dalam kondisi pencahayaan atau sudut pandang yang menantang.

**Why Calibur?**
* Membantu otomatisasi pengenalan produk ritel (Chiki Twist, Chitato, dll) secara efisien.
* Menggabungkan kecepatan *inference* TinyViT dengan antarmuka web modern berbasis Vue 3.
* Menyediakan visualisasi *bounding box* yang akurat untuk analisis stok atau *checkout*.

Sistem ini mengintegrasikan *backend* Python untuk pemrosesan AI dan *frontend* Vue.js yang responsif untuk pengalaman pengguna yang mulus.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* ![Vue.js](https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D)
* ![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)
* ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
* ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
* ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Features

🤖 **Advanced Detection:** Menggunakan TinyViT + FCOS pipeline dengan akurasi tinggi.
⚡ **Real-time Inference:** Pemrosesan gambar cepat untuk deteksi langsung dari kamera.
📱 **Responsive UI:** Dibangun dengan Vue 3 & Vite untuk performa antarmuka yang ringan.
📦 **Product Classification:** Mampu membedakan varian produk spesifik (Brand & Rasa).
📊 **Visual Feedback:** Menampilkan *confidence score* dan *bounding box* secara instan.
🇮🇩 **Indonesian Context:** Dataset dilatih khusus untuk produk pasar Indonesia.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

Ikuti langkah-langkah di bawah ini untuk menjalankan sistem Calibur di lingkungan lokal Anda.

### Prerequisites

Pastikan Anda telah menginstal perangkat lunak berikut:

* **Node.js 18+**
    ```sh
    node --version
    ```
* **Python 3.10+**
    ```sh
    python --version
    ```

### Installation

1.  **Extract the Project**
    Ekstrak file `vue-project.rar` ke direktori kerja Anda.

2.  **Set up the Frontend**
    Masuk ke folder proyek Vue dan install dependensi.
    ```sh
    cd vue-project
    npm install
    ```

3.  **Configure Environment**
    Buat file `.env` jika diperlukan untuk konfigurasi API URL.
    ```env
    VITE_API_URL=http://localhost:8000
    ```

4.  **Set up the ML Service (Backend)**
    (Asumsi Anda memiliki folder backend terpisah atau di dalam `training_code`)
    ```sh
    cd backend-service
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    pip install -r requirements.txt
    ```

5.  **Download Trained Models**
    Pastikan model berikut tersedia di direktori `models/`:
    * `Skripsi_Aug_512_from1080_alt4.rar` (Dataset/Model Utama)
    * `Skripsi_SplitThenAug_384_Threads_frontal_v5.rar` (Model Eksperimen)

6.  **Start Development Servers**
    * **Terminal 1 (Frontend):**
        ```sh
        npm run dev
        ```
    * **Terminal 2 (ML Service):**
        ```sh
        uvicorn main:app --reload --port 8000
        ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

**Basic Detection**

1.  Buka browser dan navigasi ke `http://localhost:5173`.
2.  Izinkan akses kamera jika diminta.
3.  Arahkan kamera ke produk snack target.
4.  Sistem akan menampilkan kotak deteksi (*bounding box*) dan nama produk.

**API Usage**

```javascript
// Contoh request klasifikasi gambar
const response = await fetch("http://localhost:8000/api/predict", {
  method: "POST",
  body: formData, // FormData berisi file gambar
});
const result = await response.json();
// Returns: { predictions: [{ label: "Chiki Twist", score: 0.98, box: [...] }] }
