<br />
<div align="center">
  <h3 align="center">Calibur Detection System</h3>

  <p align="center">
    Smart Snack Product Detection System for Retail Efficiency
    <br />
    ML-powered product recognition application using TinyViT + FCOS
    <br />
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
    <li><a href="#api-documentation">API Documentation</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
  </ol>
</details>

## About The Project

**Calibur** is an intelligent product detection system designed to automate the recognition of snack foods in retail environments. Developed as a thesis project, it leverages advanced Deep Learning architectures (TinyViT + FCOS) to detect and classify specific Indonesian snack brands (such as Chiki Twist, Chitato, etc.) in real-time.

**Why Calibur?**
* Retail checkout processes and inventory management often suffer from manual inefficiencies.
* Standard object detection models are often too heavy for edge deployment; Calibur optimizes this using TinyViT.
* The system combines real-time object detection with a modern web interface to provide instant feedback through visual bounding boxes, making stock monitoring faster and more accurate.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* ![Vue.js](https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D)
* ![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)
* ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
* ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
* ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Features

🤖 **ML-Powered Detection:** TinyViT + FCOS pipeline optimized for high throughput.
📱 **Responsive Design:** Modern Vue 3 interface usable on tablets and desktops.
⚡ **Real-time Inference:** Live camera feed processing with low latency.
📦 **Multi-Class Recognition:** Accurately distinguishes between similar snack packaging variants.
📊 **Visual Analytics:** Instant confidence scores and bounding box visualization.
🔒 **Local Processing:** Inference runs locally ensuring data privacy and speed.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

Follow these steps to set up Calibur locally for development.

### Prerequisites

Ensure you have the following installed:

* **Node.js 18+**
    ```sh
    node --version
    ```
* **Python 3.10+**
    ```sh
    python --version
    ```

### Installation

1.  **Extract Project Files**
    Since the source code is provided in archives, first extract `vue-project.rar` to your workspace.

2.  **Set up the Frontend**
    ```sh
    cd vue-project
    npm install
    cp .env.example .env.local
    ```

3.  **Configure Environment Variables**
    Edit `frontend/.env.local` to point to your local ML backend:
    ```env
    VITE_API_URL=http://localhost:8000
    ```

4.  **Set up the ML Service**
    Navigate to your backend/training directory:
    ```sh
    cd ../backend
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    pip install -r requirements.txt
    ```

5.  **Prepare Models & Datasets**
    Extract the model archives into the `models/` directory:
    * `Skripsi_Aug_512_from1080_alt4.rar` (Primary Model)
    * `Skripsi_SplitThenAug_384_Threads_frontal_v5.rar` (Experimental Model)

6.  **Start Development Servers**
    ```sh
    # Terminal 1 - Frontend
    npm run dev

    # Terminal 2 - ML Service
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

**Basic Detection**

1.  Navigate to `http://localhost:5173`.
2.  Click **"Start Camera"**.
3.  Allow browser camera access.
4.  Point camera at a snack product (e.g., Chiki Twist).
5.  Get instant classification with confidence scores and bounding boxes.

**API Usage**

```javascript
// Classify product image frame
const response = await fetch("http://localhost:8000/api/predict", {
  method: "POST",
  body: formData, // FormData with image file
});

const result = await response.json();
// Returns: { success: true, predictions: [ { label: "Chiki Twist", score: 0.98, box: [...] } ] }
