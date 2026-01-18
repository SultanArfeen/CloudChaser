# CloudChaser Build & Deploy Guide

## Prerequisites

Before building the app, ensure you have:

- **Node.js 18+** installed
- **Python 3.11+** with pip
- **Java JDK 17** (for Android builds)

---

## Quick Start (Development)

### 1. Start Backend API

```powershell
cd c:\Users\Arfeen\Desktop\CloudChaser\backend
pip install -r requirements.txt
python main.py
```

Backend runs at: <http://localhost:8000>

### 2. Start Frontend (Web)

```powershell
cd c:\Users\Arfeen\Desktop\CloudChaser\frontend
npm install --legacy-peer-deps
npm run dev
```

Frontend runs at: <http://localhost:3000>

---

## Building Android APK (Without Android Studio)

CloudChaser can build an APK directly using Capacitor CLI and Gradle.

### Step 1: Build the Frontend

```powershell
cd c:\Users\Arfeen\Desktop\CloudChaser\frontend
npm run build
```

This creates static files in the `out/` directory.

### Step 2: Add Android Platform

```powershell
npx cap add android
npx cap sync
```

### Step 3: Build APK with Gradle

```powershell
cd android
./gradlew assembleDebug
```

The APK will be at: `android/app/build/outputs/apk/debug/app-debug.apk`

### Step 4: Install on Device

Connect your Android device (USB debugging enabled), then:

```powershell
adb install app/build/outputs/apk/debug/app-debug.apk
```

---

## Training the ML Model

### 1. Run ETL Pipeline

First, convert the Clouds-1500 dataset to YOLO format:

```powershell
cd c:\Users\Arfeen\Desktop\CloudChaser\training
pip install -r requirements.txt
python etl_cloud1500.py --input ../CCDataset --output ./yolo_dataset
```

### 2. Train MobileNetV3

```powershell
python train_export.py --dataset ./yolo_dataset --output ./models --epochs 20
```

This will:

- Train on your GPU (if available)
- Save checkpoints to `./models/`
- Export ONNX model to `./models/cloudchaser.onnx`

### 3. Deploy Model to Frontend

Copy the trained model to the frontend:

```powershell
copy .\models\cloudchaser.onnx ..\frontend\public\models\
```

---

## Environment Notes

### Android Emulator Networking

- The app automatically uses `10.0.2.2:8000` when running on Android emulator
- This routes to your host machine's localhost

### Camera Permissions

- Android: Automatically requested at runtime
- iOS: Requires user gesture for compass permission (iOS 13+)

### Static Export Constraints

- No server-side rendering (SSR)
- No API routes in Next.js
- All data fetching is client-side
