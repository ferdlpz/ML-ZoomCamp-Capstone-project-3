# ML-ZoomCamp-Capstone-project-3

# Smartphone Rating Predictor

## Description of the Problem
The smartphone market is saturated with diverse configurations, making it difficult for consumers and manufacturers to quantify the "quality" or "value" of a device based solely on technical specifications. 

**The Objective:**
This project aims to build a supervised machine learning model to predict a smartphone's **Rating** based on its hardware and software specifications (e.g., Processor speed, RAM, Battery capacity, Camera MP, and Price). 

From a Data Science perspective, this is a **Regression task**. However, the challenge lies in:
* **Feature Engineering:** Handling categorical data like "Processor Brand" and "Operating System."
* **Data Cleaning:** Managing missing values in specialized specs (like fast charging wattage).
* **Feature Correlation:** Identifying how much "Brand Value" vs. "Technical Specs" influences the final rating.



---

## Instructions on How to Run the Project

### 1. Prerequisites
Ensure you have **Python 3.9+** installed. It is highly recommended to use a virtual environment to avoid dependency conflicts.

### 2. Installation
Clone this repository and install the required packages:

```bash
git clone <your-repo-url>
cd smartphone-rating-prediction
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

