# CartoonForge Backend

FastAPI-based backend for **AI-powered image cartoonization**.  
This server accepts images, applies cartoon-style image processing, and returns the transformed image.

> Built for learning, experimentation, and future AI model integration (CartoonGAN, FastStyle, etc.)

---

## Features

- Upload image via REST API
- Cartoon-style image processing
- FastAPI + Python
- Frontend-ready (Next.js / React / Mobile apps)
- Easily extensible to real AI models
- 100% free & open-source

---

## Tech Stack

- **Python 3.9+**
- **FastAPI**
- **Pillow (PIL)**
- **OpenCV (optional, for advanced cartoon effects)**
- **Uvicorn** (ASGI server)

---

## Project Structure

CartoonForge-backend/
│
├── main.py # FastAPI application entry point
├── requirements.txt # Python dependencies
├── .gitignore # Ignored files (venv, cache, models)
└── README.md # Project documentation

---

## Prerequisites

Make sure you have installed:

- Python **3.9 or higher**
- Git

Check versions:
```bash
python --version
git --version
```

---

## Clone the Repository

git clone https://github.com/Sanjeevan23/CartoonForge-backend.git
cd CartoonForge-backend

---

## Create Virtual Environment (Recommended)

Windows
python -m venv venv
venv\Scripts\activate

macOS / Linux
python3 -m venv venv
source venv/bin/activate

---

## Install Dependencies

pip install -r requirements.txt

---

## Run the Backend Server

uvicorn main:app --reload

You should see output like:
Uvicorn running on http://127.0.0.1:8000

---

## API Documentation (Swagger UI)

Open in browser:
http://127.0.0.1:8000/docs

This page lets you:

1. Upload images
2. Test the API
3. See request/response formats

---

## API Usage

### Endpoint
POST /enhance

### Request
Content-Type: multipart/form-data
Body: image file (file)

### Response
Returns a cartoonized image (image/png)

---

## About “AI” in This Project
Currently:
- Uses image processing (cartoon-style effects)

Planned / extendable:
- CartoonGAN
- Fast Neural Style Transfer
- Super Resolution (Real-ESRGAN)
- Multiple style presets

This backend is intentionally structured to plug in real AI models later.

---

## Author
Sanjeevan
Learning AI + Full-Stack Development

## Support
If you find this project useful:
- Star ⭐ the repo
- Fork 🍴 it
- Build something cool on top of it 🚀
