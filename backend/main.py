"""
CloudChaser Backend API
FastAPI server for cloud analysis with OpenMeteo weather data
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import httpx
import io
import base64
from PIL import Image
from datetime import datetime

app = FastAPI(
    title="CloudChaser API",
    description="Cloud classification and weather analysis service",
    version="1.0.0"
)

# CORS configuration for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "capacitor://localhost",
        "http://10.0.2.2:3000",
        "*"  # For development - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# DATA MODELS
# ============================================================================

class CloudType(BaseModel):
    id: int
    name: str
    description: str
    lwc_min: float  # Liquid Water Content min (g/m³)
    lwc_max: float  # Liquid Water Content max (g/m³)
    precipitation_risk: str
    aviation_warning: str


class WeatherData(BaseModel):
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    cloud_cover: int  # percentage
    weather_code: int
    description: str


class AnalysisResult(BaseModel):
    cloud_type: CloudType
    weather: Optional[WeatherData]
    analysis_text: str
    confidence: float
    timestamp: str


# ============================================================================
# CLOUD PHYSICS DATA
# ============================================================================

CLOUD_TYPES = {
    0: CloudType(
        id=0,
        name="Cirriform",
        description="High-altitude clouds composed of ice crystals. Thin, wispy appearance. Form above 20,000 feet.",
        lwc_min=0.01,
        lwc_max=0.05,
        precipitation_risk="None - these clouds don't produce precipitation",
        aviation_warning="Generally safe. May indicate approaching weather fronts 12-24 hours away."
    ),
    1: CloudType(
        id=1,
        name="Cumuliform",
        description="Vertically developed clouds with sharp edges. Dense, fluffy appearance. Can produce thunderstorms.",
        lwc_min=0.5,
        lwc_max=3.0,
        precipitation_risk="High - can produce heavy rain, hail, and thunderstorms",
        aviation_warning="CAUTION: Potential severe turbulence, icing, and convective activity. Avoid if developing vertically."
    ),
    2: CloudType(
        id=2,
        name="Stratiform",
        description="Layered, uniform clouds covering large areas. Flat bases with grey appearance.",
        lwc_min=0.25,
        lwc_max=0.30,
        precipitation_risk="Moderate - can produce steady light rain or drizzle",
        aviation_warning="Low visibility conditions possible. Watch for reduced ceiling heights."
    ),
    3: CloudType(
        id=3,
        name="Stratocumuliform",
        description="Hybrid clouds with rolling masses. Patchy appearance mixing stratus and cumulus features.",
        lwc_min=0.30,
        lwc_max=0.45,
        precipitation_risk="Moderate - may produce light showers",
        aviation_warning="Variable conditions. Monitor for embedded convective activity."
    ),
    4: CloudType(
        id=4,
        name="Background",
        description="Clear sky or non-cloud objects (buildings, trees, etc.)",
        lwc_min=0.0,
        lwc_max=0.0,
        precipitation_risk="None",
        aviation_warning="Clear conditions"
    )
}


# ============================================================================
# OPENMETEO API INTEGRATION
# ============================================================================

async def get_weather_data(latitude: float, longitude: float) -> Optional[WeatherData]:
    """Fetch current weather from OpenMeteo API (free, no API key required)"""
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,cloud_cover,weather_code",
        "timezone": "auto"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            
            current = data.get("current", {})
            
            # Decode weather code to description
            weather_descriptions = {
                0: "Clear sky",
                1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                45: "Fog", 48: "Depositing rime fog",
                51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
                61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
                71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
                80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
                95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
            }
            
            weather_code = current.get("weather_code", 0)
            
            return WeatherData(
                temperature=current.get("temperature_2m", 0),
                humidity=current.get("relative_humidity_2m", 0),
                pressure=current.get("surface_pressure", 1013),
                wind_speed=current.get("wind_speed_10m", 0),
                cloud_cover=current.get("cloud_cover", 0),
                weather_code=weather_code,
                description=weather_descriptions.get(weather_code, "Unknown")
            )
    except Exception as e:
        print(f"Weather API error: {e}")
        return None


def generate_analysis_text(
    cloud_type: CloudType,
    weather: Optional[WeatherData],
    confidence: float
) -> str:
    """Generate meteorological analysis text based on cloud type and weather"""
    
    analysis_parts = []
    
    # Cloud identification
    analysis_parts.append(f"**Cloud Type Identified: {cloud_type.name}**")
    analysis_parts.append(f"Classification confidence: {confidence*100:.1f}%")
    analysis_parts.append("")
    
    # Description
    analysis_parts.append(f"📋 {cloud_type.description}")
    analysis_parts.append("")
    
    # Liquid Water Content
    if cloud_type.id != 4:  # Not background
        lwc_avg = (cloud_type.lwc_min + cloud_type.lwc_max) / 2
        analysis_parts.append(f"💧 **Liquid Water Content**: {cloud_type.lwc_min:.2f} - {cloud_type.lwc_max:.2f} g/m³")
        
        if lwc_avg > 1.0:
            analysis_parts.append("   ⚠️ High water density - potential for heavy precipitation")
        elif lwc_avg > 0.3:
            analysis_parts.append("   ☁️ Moderate water content")
        else:
            analysis_parts.append("   ✓ Low water content")
    
    analysis_parts.append("")
    
    # Precipitation risk
    analysis_parts.append(f"🌧️ **Precipitation Risk**: {cloud_type.precipitation_risk}")
    analysis_parts.append("")
    
    # Weather context
    if weather:
        analysis_parts.append("📍 **Current Weather Conditions**:")
        analysis_parts.append(f"   • Temperature: {weather.temperature}°C")
        analysis_parts.append(f"   • Humidity: {weather.humidity}%")
        analysis_parts.append(f"   • Cloud Cover: {weather.cloud_cover}%")
        analysis_parts.append(f"   • Wind: {weather.wind_speed} km/h")
        analysis_parts.append(f"   • Conditions: {weather.description}")
        analysis_parts.append("")
        
        # Contextual analysis
        if weather.humidity > 80 and cloud_type.id in [1, 2, 3]:
            analysis_parts.append("⚡ High humidity combined with cloud cover suggests increased precipitation likelihood.")
    
    # Aviation warning
    if cloud_type.aviation_warning:
        analysis_parts.append(f"✈️ **Aviation Advisory**: {cloud_type.aviation_warning}")
    
    return "\n".join(analysis_parts)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {"message": "CloudChaser API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/cloud-types")
async def get_cloud_types():
    """Get all cloud type definitions"""
    return list(CLOUD_TYPES.values())


@app.get("/cloud-types/{type_id}")
async def get_cloud_type(type_id: int):
    """Get specific cloud type by ID"""
    if type_id not in CLOUD_TYPES:
        raise HTTPException(status_code=404, detail="Cloud type not found")
    return CLOUD_TYPES[type_id]


@app.get("/weather")
async def get_weather(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude")
):
    """Get current weather for a location"""
    weather = await get_weather_data(lat, lon)
    if not weather:
        raise HTTPException(status_code=503, detail="Weather service unavailable")
    return weather


@app.post("/analyze")
async def analyze_cloud(
    cloud_class: int = Query(..., ge=0, le=4, description="Predicted cloud class (0-4)"),
    confidence: float = Query(0.8, ge=0, le=1, description="Classification confidence"),
    lat: Optional[float] = Query(None, description="Latitude for weather context"),
    lon: Optional[float] = Query(None, description="Longitude for weather context"),
    file: Optional[UploadFile] = File(None, description="Cloud image (optional)")
):
    """
    Analyze a cloud classification result and provide detailed meteorological information.
    
    - **cloud_class**: Class ID from local ONNX model (0=Cirri, 1=Cumuli, 2=Strati, 3=Stratocumuli, 4=Background)
    - **confidence**: Model confidence score (0-1)
    - **lat/lon**: Location for weather context (optional but recommended)
    - **file**: The captured image (optional, for logging/future AI analysis)
    """
    
    if cloud_class not in CLOUD_TYPES:
        raise HTTPException(status_code=400, detail="Invalid cloud class")
    
    cloud_type = CLOUD_TYPES[cloud_class]
    
    # Get weather if location provided
    weather = None
    if lat is not None and lon is not None:
        weather = await get_weather_data(lat, lon)
    
    # Generate analysis
    analysis_text = generate_analysis_text(cloud_type, weather, confidence)
    
    result = AnalysisResult(
        cloud_type=cloud_type,
        weather=weather,
        analysis_text=analysis_text,
        confidence=confidence,
        timestamp=datetime.utcnow().isoformat()
    )
    
    return result


@app.get("/physics/{cloud_class}")
async def get_cloud_physics(cloud_class: int):
    """Get detailed physics data for RAG context"""
    
    if cloud_class not in CLOUD_TYPES:
        raise HTTPException(status_code=404, detail="Cloud class not found")
    
    cloud = CLOUD_TYPES[cloud_class]
    
    # Detailed physics context for RAG
    physics_data = {
        "class_id": cloud_class,
        "name": cloud.name,
        "lwc_range": f"{cloud.lwc_min}-{cloud.lwc_max} g/m³",
        "typical_altitude": {
            0: "Above 20,000 ft (6,000m) - Troposphere upper layer",
            1: "1,000-40,000 ft - Full troposphere depth",
            2: "Below 6,500 ft (2,000m) - Low level",
            3: "Below 8,000 ft (2,400m) - Low to mid level",
            4: "N/A - Not a cloud"
        }.get(cloud_class, "Unknown"),
        "formation_process": {
            0: "Ice crystal formation at very low temperatures (-40°C to -60°C)",
            1: "Strong convective updrafts lifting moist air rapidly",
            2: "Widespread lifting of stable air over large area",
            3: "Convective elements within stable layer",
            4: "N/A"
        }.get(cloud_class, "Unknown"),
        "solar_attenuation": {
            0: "Low (10-20%) - High transparency",
            1: "High (60-90%) - Dense, causes rapid solar ramps",
            2: "Moderate (40-60%) - Consistent diffuse light",
            3: "Variable (30-70%) - Patchy attenuation",
            4: "None - Clear"
        }.get(cloud_class, "Unknown"),
        "precipitation_physics": cloud.precipitation_risk,
        "aviation_considerations": cloud.aviation_warning
    }
    
    return physics_data


# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    print("="*50)
    print("CloudChaser API Starting...")
    print("="*50)
    print("Endpoints:")
    print("  GET  /              - API info")
    print("  GET  /health        - Health check")
    print("  GET  /cloud-types   - List all cloud types")
    print("  GET  /weather       - Get weather data")
    print("  POST /analyze       - Analyze cloud classification")
    print("  GET  /physics/{id}  - Get cloud physics data")
    print("="*50)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
