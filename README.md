
# 🧠 Face Verification System (Offline | FastAPI | Docker)

A modular, offline-ready face verification service built using FastAPI and InsightFace.  
It compares two facial images — either a **selfie and an ID photo**, or two selfies — and returns a **match score**, **confidence level**, and **verification decision**.

---

## 🚀 Features

- 🔒 100% offline — no external APIs or cloud dependency
- 🧠 Powered by [InsightFace](https://github.com/deepinsight/insightface)'s `buffalo_l` model
- 🧼 Advanced preprocessing for noisy ID images (CLAHE, gamma, denoising)
- ⚖️ Dynamic thresholding & adaptive scoring
- 🔧 Fully configurable via `config.json`
- 🐳 Production-ready Docker deployment
- 🧪 Automated test script with real and edge-case images

---

## 📁 Project Structure

```
face-verification/
├── main.py                # FastAPI application
├── face_matcher.py        # Model loading, embedding, scoring logic
├── preprocess.py          # Image enhancement pipeline
├── config.json            # Thresholds, filters, and settings
├── download_models.py     # Downloads InsightFace model
├── test_api.py            # End-to-end API test script
├── Dockerfile             # Docker build file
├── requirements.txt       # Python dependencies
├── models/                # Local storage of InsightFace models
└── sample_images/         # Selfies, ID photos, edge cases
```

---

## ⚙️ Setup & Installation

### 🔧 Local (Dev)
```bash
# Clone repo
git clone https://github.com/your-username/face-verification.git
cd face-verification

# (Optional) Create virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download model
python download_models.py

# Run API
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

### 🐳 Docker
```bash
# Build Docker image
docker build -t face-verifier .

# Run the container
docker run -p 8000:8000 face-verifier
```

---

## 🔍 API Endpoints

| Method | Endpoint        | Description                          |
|--------|------------------|--------------------------------------|
| POST   | `/verify-face`   | Accepts 2 base64 images, returns match |
| GET    | `/health`        | Health check                         |
| GET    | `/version`       | Model/version info                   |
| GET    | `/config`        | Current configuration                |

📌 View full Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Testing

```bash
# Run test script (validates API with sample images)
python test_api.py
```

✔ Tests include:
- Matching selfie vs selfie
- Matching ID vs selfie
- Non-match cases
- Edge cases (blur, occlusion, no face)

---

## ⚙️ Configuration (`config.json`)

Customizable options:
- `thresholds`: different for `id_to_selfie` and `selfie_to_selfie`
- `preprocessing`: enables contrast enhancement, denoising, gamma, etc.
- `confidence_thresholds`: maps scores to low/medium/high labels
- `detection`: face detection size, max faces
- `logging`: logging level and format

---

## 🧠 Model Used

- Model: `buffalo_l` from InsightFace
- Functionality:
  - Face detection
  - 512-D face embedding
- Auto GPU fallback via ONNX Runtime

---

## ✅ Sample Output

```json
{
  "match_score": 0.8432,
  "match": true,
  "confidence_level": "high",
  "threshold": 0.80,
  "status": "success"
}
```

---

## 📌 Future Improvements

- Liveness detection support
- Face alignment & landmark visualization
- Admin dashboard for logs & analytics
- Multi-face support (batch API)

---

## 🛡 License

MIT © [Your Name]  
Uses open-source models from [InsightFace](https://github.com/deepinsight/insightface).
