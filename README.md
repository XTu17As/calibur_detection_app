<div align="center">
Calibur
Aplikasi Frontend Modern Berbasis Vue 3
Vue 3
Vite
Pinia
License

Bangun aplikasi web yang cepat, scalable, dan mudah dipelihara dengan stack teknologi terkini

🚀 Demo • 📖 Dokumentasi • 🎯 Features • 💻 Quick Start

</div>
🌟 Tentang Calibur
Calibur adalah aplikasi frontend modern yang dibangun dengan Vue 3 Composition API, dirancang untuk memberikan pengalaman pengembangan yang optimal dan performa aplikasi yang luar biasa. Dengan memanfaatkan ekosistem Vue.js terbaru dan tooling generasi baru, Calibur siap menjadi fondasi kuat untuk proyek web Anda.

✨ Fitur Utama
<table> <tr> <td width="50%">
⚡ Lightning Fast
Hot Module Replacement (HMR) instan

Build time yang sangat efisien

Optimasi bundle otomatis

</td> <td width="50%">
🎨 Modern Architecture
Vue 3 Composition API

Type-safe dengan Vite

Modular component structure

</td> </tr> <tr> <td width="50%">
🔄 State Management
Pinia untuk state global

Store yang reactive dan modular

DevTools integration

</td> <td width="50%">
🧪 Testing Ready
Vitest framework terintegrasi

Unit testing support

Component testing tools

</td> </tr> </table>
🛠️ Tech Stack
text
graph LR
    A[Vue 3] --> B[Vite]
    B --> C[Pinia]
    C --> D[Vue Router]
    D --> E[Vitest]
    style A fill:#4FC08D
    style B fill:#646CFF
    style C fill:#FFD859
    style D fill:#41B883
    style E fill:#729B1B
Teknologi	Versi	Deskripsi
🟢 Vue 3	^3.x	Progressive JavaScript framework dengan Composition API
⚡ Vite	^5.x	Next-generation frontend tooling
🍍 Pinia	^2.x	Intuitive state management untuk Vue
🛣️ Vue Router	^4.x	Official router untuk Single Page Applications
🧪 Vitest	^1.x	Blazing fast unit testing framework
📂 Struktur Proyek
text
calibur/
│
├── 📁 src/
│   ├── 🎨 assets/          # Gambar, ikon, dan stylesheet
│   ├── 🧩 components/       # Reusable UI components
│   ├── 📄 views/            # Halaman utama aplikasi
│   ├── 🗄️ stores/           # Pinia state management
│   ├── 🛤️ router/           # Konfigurasi routing
│   ├── 🔧 utils/            # Helper functions & utilities
│   ├── 🎯 App.vue           # Root component
│   └── 🚀 main.js           # Entry point aplikasi
│
├── 📁 public/              # Static assets
├── 📁 tests/               # Test files
├── 📋 package.json         # Dependencies & scripts
├── ⚙️ vite.config.js       # Vite configuration
└── 📖 README.md            # You are here!
💻 Quick Start
Prasyarat
Pastikan sistem Anda memiliki:

📦 Node.js ≥ 18.x (LTS recommended)

📥 npm atau yarn atau pnpm

🎯 Langkah Instalasi
1️⃣ Ekstraksi Project
bash
# Ekstrak file vue-project.rar ke direktori pilihan Anda
# Contoh lokasi: ~/projects/calibur
2️⃣ Install Dependencies
bash
# Masuk ke direktori project
cd calibur

# Install semua dependencies
npm install
3️⃣ Jalankan Development Server
bash
# Start dev server dengan HMR
npm run dev
🎉 Aplikasi akan berjalan di http://localhost:5173

📜 Available Scripts
Command	Deskripsi
npm run dev	🚀 Menjalankan development server
npm run build	📦 Build aplikasi untuk production
npm run preview	👀 Preview production build
npm run test	🧪 Menjalankan unit tests
npm run lint	🔍 Check code quality
🎨 Kustomisasi
Tema & Styling
Anda dapat mengkustomisasi tema aplikasi dengan mengedit file di src/assets/:

javascript
// src/assets/theme.js
export default {
  colors: {
    primary: '#4FC08D',
    secondary: '#646CFF',
    accent: '#FFD859'
  }
}
Environment Variables
Buat file .env untuk konfigurasi environment:

text
VITE_APP_TITLE=Calibur
VITE_API_BASE_URL=https://api.example.com
VITE_APP_VERSION=1.0.0
🤝 Contributing
Kontribusi sangat diterima! Silakan ikuti langkah berikut:

🍴 Fork repository ini

🌿 Buat branch fitur (git checkout -b feature/AmazingFeature)

✍️ Commit perubahan (git commit -m 'Add some AmazingFeature')

📤 Push ke branch (git push origin feature/AmazingFeature)

🎉 Buat Pull Request

📝 License
Distributed under the MIT License. See LICENSE for more information.

📬 Contact & Support
Jika Anda memiliki pertanyaan atau butuh bantuan:

💬 Discord: Join our community

📧 Email: support@calibur.dev

🐛 Issues: Report bugs

<div align="center">
⭐ Don't forget to give this project a star if you found it helpful!
Made with ❤️ using Vue 3

⬆ Back to Top

</div>
