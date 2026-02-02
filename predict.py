import pickle

from flask import Flask, request, jsonify
import pandas as pd

# functions clean
def parse_cameras(row):
    try:
        # convert string to real list using ast.literal_eval
        cam_list = ast.literal_eval(row)
        # convert each element to float
        cam_list = [float(c) for c in cam_list]
        return pd.Series([cam_list[0], sum(cam_list)])
    except:
        return pd.Series([0, 0])
    
def extract_pixels(res_string):
    try:
        # Buscamos los números y los multiplicamos
        parts = res_string.lower().split('x')
        width = int(''.join(filter(str.isdigit, parts[0])))
        height = int(''.join(filter(str.isdigit, parts[1])))
        return width * height
    except:
        return 0

def clean_df(df_raw):
    df = df_raw.copy()
    df['brand'] = df['model'].str.split(' ').str[0]

    cat_cols_with_nan = ['os', 'core_type', 'display_type', 'memory_card_type']
    for col in cat_cols_with_nan:
        # if the brand dont has mode, we fill with overall mode
        df[col] = df.groupby('brand')[col].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else df[col].mode()[0]))

    num_cols_to_group = ['clock_ghz', 'battery_mah', 'front_camera_mp', 'rear_camera_max_mp']
    for col in num_cols_to_group:
        df[col] = df.groupby('brand')[col].transform(lambda x: x.fillna(x.median() if x.notna().any() else df[col].median()))

    df['fast_charge_w'] = df['fast_charge_w'].fillna(0)
    df['refresh_rate_hz'] = df['refresh_rate_hz'].fillna(60)

    # 3. Feature Extraction 
    # (Aquí pegas las funciones parse_cameras y extract_pixels)
    df[['primary_rear_mp', 'total_rear_mp']] = df['rear_camera_mp_list'].apply(parse_cameras)
    df['total_pixels'] = df['resolution'].apply(extract_pixels)
    
    feature_selection = [
    'price', 'storage_gb', 
    'battery_mah', 'fast_charge_w', 'screen_size_in', 
    'refresh_rate_hz', 'primary_rear_mp', 'total_pixels', 
    'front_camera_mp', 'memory_card_max_gb', 'rear_camera_count', 
    'network_type',  'memory_card_type', 'brand'
    ]

    return df[feature_selection]

model_file = 'model_n_estimators=500_max_depth=6_learning_rate=0.01.bin'

import os
os.chdir('/Users/fdl/Repos/ML-ZoomCamp-Capstone-project-3/')

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('rating-predict')

@app.route('/predict', methods=['POST'])
def predict():
    cell_phone = request.get_json()
    df_cell = pd.DataFrame([cell_phone])
    X = clean_df(df_cell)
    prediction = model.predict(X)[0]
    
    result = {
        'cell_score_prediction': float(round(prediction, 2))
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)