# ML-ZoomCamp-Capstone-project-3

# Smartphone Rating Predictor

## Description of the Problem
The smartphone market is highly competitive, with hundreds of devices launched annually. For both consumers and manufacturers, it is challenging to objectively quantify a phone's "quality" or "market rating" based solely on technical specifications. 

**The Objective:**
This project aims to build a robust machine learning regression model to predict a smartphone's **Rating** (0-100) using its technical specifications (Price, RAM, Battery, Processor Clock Speed, etc.).

**Key Technical Challenges:**
* **Feature Engineering:** We extracted meaningful signals from raw text, such as calculating "Total Pixels" from screen resolutions and parsing "Primary Camera Megapixels" from complex list strings.
* **Data Imputation:** Developed a domain-specific imputation strategy (e.g., assuming 60Hz for missing refresh rates or 0W for missing fast charge specs).
* **Modeling:** Evaluated Linear Regression, Random Forest, and XGBoost. The final model is an **XGBoost Regressor** optimized via Grid Search and bundled into a Scikit-Learn Pipeline for seamless deployment.

---

## Instructions on How to Run the Project

### 1. Prerequisites
- **Python 3.11+**
- **Docker** (for containerized deployment)
- **uv** (high-performance Python package manager)

### 2. Environment Setup
We use `uv` for lightning-fast and reproducible dependency management.
```bash
# Install uv if you haven't
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Synchronize dependencies
uv sync

--
```

## Model Training
To clean the data, perform feature engineering, and train the final XGBoost pipeline:
python train.py

## Running the API (Flask)
You can run the prediction service locally:

```bash
python predict.py
```
The service will be available at http://0.0.0.0:9696

## Deployment with Docker
To ensure the environment is identical regardless of the host machine, use the provided Dockerfile:

# Build the image
```bash
docker build -t smartphone-rating-service .
```

# Run the container
```bash
docker run -it -p 9696:9696 smartphone-rating-service
```

## Testing the Service
You can test the running API by sending a JSON payload via `curl`:
```bash
curl -X POST http://localhost:9696/predict \
     -H "Content-Type: application/json" \
     -d '{
        "model": "Samsung Galaxy S23",
        "price": 850,
        "storage_gb": 256,
        "battery_mah": 3900,
        "fast_charge_w": 25,
        "screen_size_in": 6.1,
        "refresh_rate_hz": 120,
        "rear_camera_mp_list": "[50, 10, 12]",
        "resolution": "1080 x 2340",
        "front_camera_mp": 12,
        "memory_card_max_gb": 0,
        "rear_camera_count": 3,
        "network_type": "5G",
        "memory_card_type": "None"
     }'
```