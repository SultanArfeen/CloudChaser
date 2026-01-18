'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Capacitor } from '@capacitor/core';
import {
    analyzeCloud,
    OFFLINE_CLOUD_TYPES,
    type CloudType,
    type AnalysisResult,
    type WeatherData
} from '@/lib/api';
import {
    loadModel,
    classifyImage,
    isModelLoaded,
    CLASS_NAMES,
    type InferenceResult
} from '@/lib/inference';
import { useDeviceOrientation, getCardinalDirection } from '@/lib/useDeviceOrientation';

// Icons as inline SVGs
const CameraIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="13" r="3" />
        <path d="M5 7h2l2-2h6l2 2h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2z" />
    </svg>
);

const UploadIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
);

const CloseIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <line x1="18" y1="6" x2="6" y2="18" />
        <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
);

const InfoIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="16" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
);

export default function HomePage() {
    // State
    const [isLoading, setIsLoading] = useState(true);
    const [modelReady, setModelReady] = useState(false);
    const [currentClass, setCurrentClass] = useState<number>(4);
    const [confidence, setConfidence] = useState<number>(0);
    const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
    const [showAnalysis, setShowAnalysis] = useState(false);
    const [weather, setWeather] = useState<WeatherData | null>(null);
    const [location, setLocation] = useState<{ lat: number, lon: number } | null>(null);
    const [statusMessage, setStatusMessage] = useState<string>('');
    const [isCapturing, setIsCapturing] = useState(false);
    const [uploadedImage, setUploadedImage] = useState<string | null>(null);
    const [showUploadPreview, setShowUploadPreview] = useState(false);

    // Refs
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const uploadedImgRef = useRef<HTMLImageElement>(null);
    const inferenceIntervalRef = useRef<NodeJS.Timeout | null>(null);

    // Hooks
    const { orientation, permissionState, requestPermission } = useDeviceOrientation();

    // Get cloud type info
    const cloudType = OFFLINE_CLOUD_TYPES[currentClass];
    const lwcAvg = (cloudType.lwc_min + cloudType.lwc_max) / 2;

    // Show status message temporarily
    const showStatus = useCallback((message: string, duration = 3000) => {
        setStatusMessage(message);
        setTimeout(() => setStatusMessage(''), duration);
    }, []);

    // Initialize camera
    useEffect(() => {
        let stream: MediaStream | null = null;

        async function initCamera() {
            try {
                // Check if running in Capacitor
                if (Capacitor.isNativePlatform()) {
                    // Use Capacitor Camera Preview for native
                    const { CameraPreview } = await import('@capacitor-community/camera-preview');

                    // Add transparency class to body
                    document.body.classList.add('camera-active');

                    await CameraPreview.start({
                        parent: 'camera-container',
                        position: 'rear',
                        toBack: true,
                        width: window.innerWidth,
                        height: window.innerHeight,
                        enableZoom: true,
                    });

                    setIsLoading(false);
                    showStatus('Camera ready');
                } else {
                    // Web fallback - use getUserMedia
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: {
                            facingMode: 'environment',
                            width: { ideal: 1280 },
                            height: { ideal: 720 }
                        }
                    });

                    if (videoRef.current) {
                        videoRef.current.srcObject = stream;
                        await videoRef.current.play();
                    }

                    setIsLoading(false);
                    showStatus('Camera ready (web mode)');
                }
            } catch (error) {
                console.error('Camera initialization failed:', error);
                setIsLoading(false);
                showStatus('No camera - use Upload button');
            }
        }

        initCamera();

        return () => {
            // Cleanup
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            document.body.classList.remove('camera-active');

            if (Capacitor.isNativePlatform()) {
                import('@capacitor-community/camera-preview').then(({ CameraPreview }) => {
                    CameraPreview.stop();
                });
            }
        };
    }, [showStatus]);

    // Load ML model
    useEffect(() => {
        async function initModel() {
            try {
                await loadModel('/models/cloudchaser.onnx');
                setModelReady(true);
                showStatus('AI model loaded');
            } catch (error) {
                console.error('Model loading failed:', error);
                // Continue without model - will use backend-only mode
                showStatus('AI model not available - using cloud mode');
            }
        }

        initModel();
    }, [showStatus]);

    // Get user location
    useEffect(() => {
        if ('geolocation' in navigator) {
            navigator.geolocation.getCurrentPosition(
                (pos) => {
                    setLocation({
                        lat: pos.coords.latitude,
                        lon: pos.coords.longitude
                    });
                },
                (error) => {
                    console.log('Location unavailable:', error);
                }
            );
        }
    }, []);

    // Real-time inference loop
    useEffect(() => {
        if (!modelReady || !videoRef.current) return;

        const runInference = async () => {
            try {
                if (videoRef.current && videoRef.current.readyState >= 2) {
                    const result = await classifyImage(videoRef.current);
                    setCurrentClass(result.classId);
                    setConfidence(result.confidence);
                }
            } catch (error) {
                // Silently fail - inference is best-effort
            }
        };

        // Run inference every 500ms
        inferenceIntervalRef.current = setInterval(runInference, 500);

        return () => {
            if (inferenceIntervalRef.current) {
                clearInterval(inferenceIntervalRef.current);
            }
        };
    }, [modelReady]);

    // Handle file upload
    const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            const dataUrl = e.target?.result as string;
            setUploadedImage(dataUrl);
            setShowUploadPreview(true);
            showStatus('Image loaded - analyzing...');
        };
        reader.readAsDataURL(file);
    };

    // Analyze uploaded image
    const analyzeUploadedImage = async () => {
        if (!uploadedImgRef.current || !modelReady) {
            showStatus('Model not ready yet');
            return;
        }

        setIsCapturing(true);
        showStatus('Analyzing uploaded image...');

        try {
            // Run inference on uploaded image
            const result = await classifyImage(uploadedImgRef.current);
            setCurrentClass(result.classId);
            setConfidence(result.confidence);

            // Get analysis from backend
            const analysisData = await analyzeCloud(
                result.classId,
                result.confidence,
                location?.lat,
                location?.lon
            );

            setAnalysisResult(analysisData);
            setWeather(analysisData.weather);
            setShowAnalysis(true);
            setShowUploadPreview(false);
        } catch (error) {
            console.error('Analysis failed:', error);

            // Offline fallback
            const offlineResult: AnalysisResult = {
                cloud_type: OFFLINE_CLOUD_TYPES[currentClass],
                weather: null,
                analysis_text: `**Cloud Type: ${CLASS_NAMES[currentClass]}**\n\nConfidence: ${(confidence * 100).toFixed(1)}%\n\n${OFFLINE_CLOUD_TYPES[currentClass].description}\n\n💧 LWC: ${OFFLINE_CLOUD_TYPES[currentClass].lwc_min} - ${OFFLINE_CLOUD_TYPES[currentClass].lwc_max} g/m³\n\n⚠️ Offline mode - connect to backend for full analysis.`,
                confidence,
                timestamp: new Date().toISOString()
            };

            setAnalysisResult(offlineResult);
            setShowAnalysis(true);
            setShowUploadPreview(false);
            showStatus('Offline analysis');
        } finally {
            setIsCapturing(false);
        }
    };

    // Capture and analyze from camera
    const handleCapture = async () => {
        if (isCapturing) return;

        setIsCapturing(true);
        showStatus('Analyzing cloud...');

        try {
            // Get analysis from backend
            const result = await analyzeCloud(
                currentClass,
                confidence,
                location?.lat,
                location?.lon
            );

            setAnalysisResult(result);
            setWeather(result.weather);
            setShowAnalysis(true);
        } catch (error) {
            console.error('Analysis failed:', error);

            // Offline fallback
            const offlineResult: AnalysisResult = {
                cloud_type: OFFLINE_CLOUD_TYPES[currentClass],
                weather: null,
                analysis_text: `**Cloud Type: ${CLASS_NAMES[currentClass]}**\n\nConfidence: ${(confidence * 100).toFixed(1)}%\n\n${OFFLINE_CLOUD_TYPES[currentClass].description}\n\n💧 LWC: ${OFFLINE_CLOUD_TYPES[currentClass].lwc_min} - ${OFFLINE_CLOUD_TYPES[currentClass].lwc_max} g/m³\n\n⚠️ Offline mode - connect to backend for full analysis.`,
                confidence,
                timestamp: new Date().toISOString()
            };

            setAnalysisResult(offlineResult);
            setShowAnalysis(true);
            showStatus('Offline analysis');
        } finally {
            setIsCapturing(false);
        }
    };

    // Trigger file input
    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    // Get risk level
    const getRiskLevel = (): 'low' | 'moderate' | 'high' => {
        if (currentClass === 1) return 'high'; // Cumuliform
        if (currentClass === 2 || currentClass === 3) return 'moderate';
        return 'low';
    };

    // Get class-specific CSS class
    const getClassCSS = (): string => {
        const classes = ['cirriform', 'cumuliform', 'stratiform', 'stratocumuliform', 'background'];
        return classes[currentClass] || 'background';
    };

    return (
        <div className="app-container">
            {/* Hidden file input for upload */}
            <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                style={{ display: 'none' }}
            />

            {/* Camera Container */}
            <div id="camera-container" className="camera-container">
                {!Capacitor.isNativePlatform() && (
                    <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        style={{
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover',
                        }}
                    />
                )}
            </div>

            {/* Hidden canvas for image processing */}
            <canvas ref={canvasRef} style={{ display: 'none' }} />

            {/* AR Overlay */}
            <div className="ar-overlay">
                {/* Top Bar */}
                <div className="top-bar">
                    {/* Compass */}
                    <div className="compass-container glass-card">
                        <div className="compass">
                            <div
                                className="compass-needle"
                                style={{
                                    transform: `translateX(-50%) translateY(-100%) rotate(${orientation.heading || 0}deg)`
                                }}
                            />
                            <div className="compass-center" />
                            <span className="compass-direction north">N</span>
                            <span className="compass-direction east">E</span>
                            <span className="compass-direction south">S</span>
                            <span className="compass-direction west">W</span>
                        </div>
                    </div>

                    {/* Cloud Classification */}
                    <div className="glass-card" style={{ padding: '12px 16px' }}>
                        <div className={`cloud-label ${getClassCSS()}`}>
                            {CLASS_NAMES[currentClass]}
                        </div>
                        <div className="confidence-meter">
                            <div className="confidence-bar">
                                <div
                                    className="confidence-fill"
                                    style={{ width: `${confidence * 100}%` }}
                                />
                            </div>
                            <span>{(confidence * 100).toFixed(0)}%</span>
                        </div>
                        <div style={{ marginTop: '8px' }}>
                            <span className={`risk-badge ${getRiskLevel()}`}>
                                {getRiskLevel().toUpperCase()} RISK
                            </span>
                        </div>
                    </div>
                </div>

                {/* Spacer */}
                <div style={{ flex: 1 }} />

                {/* Bottom Panel */}
                <div className="bottom-panel">
                    <div className="glass-card" style={{ padding: '16px' }}>
                        {/* LWC Display */}
                        <div className="lwc-display">
                            <span className="lwc-label">Liquid Water Content</span>
                            <div className="lwc-value">
                                {currentClass !== 4 ? (
                                    <>
                                        {lwcAvg.toFixed(2)}
                                        <span className="lwc-unit">g/m³</span>
                                    </>
                                ) : (
                                    <span style={{ color: 'var(--color-text-muted)', fontSize: '1rem' }}>
                                        Clear / No Cloud
                                    </span>
                                )}
                            </div>
                        </div>

                        {/* Weather Strip */}
                        {weather && (
                            <div className="weather-strip">
                                <div className="weather-item">
                                    🌡️ <span className="weather-value">{weather.temperature}°C</span>
                                </div>
                                <div className="weather-item">
                                    💧 <span className="weather-value">{weather.humidity}%</span>
                                </div>
                                <div className="weather-item">
                                    🧭 <span className="weather-value">{getCardinalDirection(orientation.heading)}</span>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Action Buttons */}
                    <div className="action-bar">
                        {/* Upload Button */}
                        <button
                            className="action-button secondary"
                            onClick={handleUploadClick}
                            title="Upload Image"
                        >
                            <UploadIcon />
                        </button>

                        {/* Camera Capture Button */}
                        <button
                            className="action-button primary"
                            onClick={handleCapture}
                            disabled={isCapturing}
                        >
                            {isCapturing ? (
                                <div className="spinner" style={{ width: '24px', height: '24px' }} />
                            ) : (
                                <CameraIcon />
                            )}
                        </button>
                    </div>
                </div>
            </div>

            {/* Status Message */}
            {statusMessage && (
                <div className="status-message info">
                    {statusMessage}
                </div>
            )}

            {/* Loading Overlay */}
            {isLoading && (
                <div className="permission-overlay">
                    <div className="spinner" style={{ width: '60px', height: '60px', marginBottom: '20px' }} />
                    <h2>Initializing CloudChaser</h2>
                    <p style={{ color: 'var(--color-text-muted)' }}>
                        Setting up camera and AI model...
                    </p>
                </div>
            )}

            {/* iOS Compass Permission */}
            {permissionState === 'prompt' && !isLoading && (
                <div style={{
                    position: 'fixed',
                    bottom: '120px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    zIndex: 60,
                }}>
                    <button
                        className="permission-button"
                        onClick={requestPermission}
                        style={{ background: 'var(--glass-bg)', border: '1px solid var(--glass-border)' }}
                    >
                        🧭 Enable Compass
                    </button>
                </div>
            )}

            {/* Upload Preview Modal */}
            {showUploadPreview && uploadedImage && (
                <div className="analysis-modal" onClick={() => setShowUploadPreview(false)}>
                    <div className="analysis-content glass-card" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '90vw' }}>
                        <div className="analysis-header">
                            <h2 className="analysis-title">
                                📷 Analyze Image
                            </h2>
                            <button className="close-button" onClick={() => setShowUploadPreview(false)}>
                                <CloseIcon />
                            </button>
                        </div>
                        <div style={{ marginBottom: '16px', borderRadius: '8px', overflow: 'hidden' }}>
                            <img
                                ref={uploadedImgRef}
                                src={uploadedImage}
                                alt="Uploaded cloud"
                                style={{ width: '100%', maxHeight: '300px', objectFit: 'contain' }}
                                crossOrigin="anonymous"
                            />
                        </div>
                        <button
                            className="permission-button"
                            onClick={analyzeUploadedImage}
                            disabled={isCapturing || !modelReady}
                            style={{ width: '100%', padding: '16px', fontSize: '1.1rem' }}
                        >
                            {isCapturing ? '🔄 Analyzing...' : modelReady ? '🔍 Analyze This Cloud' : '⏳ Loading Model...'}
                        </button>
                    </div>
                </div>
            )}

            {/* Analysis Modal */}
            {showAnalysis && analysisResult && (
                <div className="analysis-modal" onClick={() => setShowAnalysis(false)}>
                    <div className="analysis-content glass-card" onClick={(e) => e.stopPropagation()}>
                        <div className="analysis-header">
                            <h2 className="analysis-title">
                                ☁️ Cloud Analysis
                            </h2>
                            <button className="close-button" onClick={() => setShowAnalysis(false)}>
                                <CloseIcon />
                            </button>
                        </div>
                        <div
                            className="analysis-text"
                            dangerouslySetInnerHTML={{
                                __html: analysisResult.analysis_text
                                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                                    .replace(/\n/g, '<br />')
                            }}
                        />
                    </div>
                </div>
            )}
        </div>
    );
}
