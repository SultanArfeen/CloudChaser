# CloudChaser ☁️

An AI-powered Android app that identifies the weather by looking at the clouds through your camera lens.

## Features

- **Real-time Cloud Classification**: 5 cloud types (Cirriform, Cumuliform, Stratiform, Stratocumuliform, Background)
- **On-Device ML**: MobileNetV3-Small with ONNX Runtime Web (WebGL acceleration)
- **AR Camera Interface**: Live overlay with compass and classification results
- **Weather Integration**: OpenMeteo API for temperature, humidity, wind data
- **Liquid Water Content (LWC)**: Estimates cloud density and precipitation risk
- **Offline Support**: Works without network for basic classification

## Tech Stack

- **Frontend**: Next.js 15, React 19, TypeScript, Capacitor
- **Backend**: FastAPI, Python 3.12+
- **ML**: PyTorch, MobileNetV3-Small, ONNX Runtime
- **Mobile**: Capacitor (Android), Camera Preview plugin
- **Weather**: OpenMeteo API (free, no key required)

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.12+ (for ML training)
- Java JDK 17 (for Android builds)

### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Backend runs at: <http://localhost:8000>

### 2. Frontend Development

```bash
cd frontend
npm install --legacy-peer-deps
npm run dev
```

Frontend runs at: <http://localhost:3000>

### 3. Train ML Model (Optional)

```bash
cd training
pip install -r requirements.txt

# Convert dataset
python etl_cloud1500.py --input ../CCDataset --output ./yolo_dataset

# Train model (requires GPU)
python train_export.py --dataset ./yolo_dataset --epochs 20

# Copy model to frontend
copy models\cloudchaser.onnx ..\frontend\public\models\
```

### 4. Build Android APK

```bash
cd frontend
npm run build
npx cap add android
npx cap sync
cd android
.\gradlew assembleDebug
```

APK location: `android/app/build/outputs/apk/debug/app-debug.apk`

## Project Structure

```
CloudChaser/
├── backend/              # FastAPI server
│   ├── main.py          # API endpoints
│   └── requirements.txt
├── frontend/             # Next.js 15 + Capacitor
│   ├── src/app/         # React pages
│   ├── src/lib/         # API client, ONNX inference
│   └── capacitor.config.ts
├── training/             # ML pipeline
│   ├── etl_cloud1500.py  # Dataset conversion
│   └── train_export.py   # Model training
└── BUILD.md              # Detailed build guide
```

## Dataset

The app uses the **Clouds-1500** dataset from UFSC (Federal University of Santa Catarina):

- 1500 ground-based horizon images
- 5 classes with Supervisely polygon annotations
- Converted to YOLO format for training

## Architecture

### Split-Brain Inference

1. **Edge (On-Device)**: ONNX model runs in browser via WebGL for instant classification
2. **Cloud (Backend)**: FastAPI provides weather context and detailed analysis

### Key Components

- **ETL Pipeline**: Converts Supervisely JSON → YOLO bounding boxes
- **MobileNetV3**: Lightweight CNN optimized for mobile
- **ONNX Export**: Opset 14 for Hardsigmoid/Hardswish support
- **Transparent Camera**: CSS hack for Capacitor Camera Preview
- **Compass**: Device orientation with iOS 13+ permission handling

## License

MIT

## Credits

- Dataset: Clouds-1500 (UFSC, Brazil)
- Weather API: OpenMeteo
- ML Framework: PyTorch, ONNX Runtime
- Mobile: Capacitor
