<div align="center">

  <h1>⚔️ C A L I B U R ⚔️</h1>
  
  <p>
    <strong>Aplikasi Frontend Berbasis Vue 3 dengan Arsitektur Modern</strong>
  </p>

  <p>
    <a href="https://vuejs.org/">
      <img src="https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D" alt="Vue 3" />
    </a>
    <a href="https://vitejs.dev/">
      <img src="https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite" />
    </a>
    <a href="https://pinia.vuejs.org/">
      <img src="https://img.shields.io/badge/Pinia-Map_State-FFD11B?style=for-the-badge&logo=pinia&logoColor=black" alt="Pinia" />
    </a>
  </p>

  <p>
    <i>"Ditempa dengan kode, dipertajam oleh Vite."</i>
  </p>

  <br />

  <p align="center">
    <a href="#-fitur-unggulan">Fitur</a> •
    <a href="#-gudang-senjata-teknologi">Teknologi</a> •
    <a href="#-peta-wilayah-struktur">Struktur</a> •
    <a href="#-ritual-instalasi">Instalasi</a>
  </p>
</div>

<hr />

## 📖 Tentang Proyek

**Calibur** bukan sekadar antarmuka pengguna; ini adalah manifestasi dari pengembangan web modern. Dibangun di atas fondasi kokoh **Vue 3**, proyek ini dirancang untuk kecepatan, skalabilitas, dan pengalaman pengembang (*DX*) yang superior.

Kode sumber ini menggunakan *tooling* standar **Vite**, memastikan waktu *build* secepat kilat dan *Hot Module Replacement* (HMR) yang instan. Siap untuk dikembangkan, siap untuk di-deploy.

---

## 🛡️ Gudang Senjata (Teknologi)

Kami menggunakan serangkaian teknologi mutakhir untuk memastikan performa yang tak tertandingi:

| Teknologi | Lencana | Deskripsi |
| :--- | :---: | :--- |
| **Vue 3** | <img src="https://img.shields.io/badge/Core-Vue_3-4FC08D" /> | Menggunakan *Composition API* untuk logika yang modular. |
| **Vite** | <img src="https://img.shields.io/badge/Build-Vite-646CFF" /> | *Build tool* generasi terbaru untuk HMR super cepat. |
| **Pinia** | <img src="https://img.shields.io/badge/State-Pinia-FFD11B" /> | Manajemen *state* intuitif, pengganti spiritual Vuex. |
| **Vue Router** | <img src="https://img.shields.io/badge/Nav-Router-35495E" /> | Navigasi SPA (*Single Page Application*) yang mulus. |
| **Vitest** | <img src="https://img.shields.io/badge/Test-Vitest-729B1B" /> | Pengujian unit yang terintegrasi penuh dengan Vite. |

---

## 🗺️ Peta Wilayah (Struktur)

Struktur direktori disusun dengan rapi agar Anda tidak tersesat dalam kode:

```fs
📂 calibur-project
├── 📂 public           # Berkas statis publik
├── 📂 src
│   ├── 📂 assets       # 🎨 Gambar, font, dan gaya global
│   ├── 📂 components   # 🧩 Komponen UI yang dapat digunakan kembali (LEGO bricks)
│   ├── 📂 views        # 🖼️ Halaman utama yang dirender Router
│   ├── 📂 stores       # 💾 Lumbung data global (Pinia Store)
│   ├── 📂 router       # 🧭 Kompas navigasi aplikasi
│   ├── 📂 utils        # 🛠️ Fungsi pembantu & logika umum
│   ├── 📜 App.vue      # Akar dari segala komponen
│   └── 📜 main.js      # Titik masuk aplikasi
├── 📜 index.html       # Kanvas utama
└── 📜 vite.config.js   # Konfigurasi dapur pacu
