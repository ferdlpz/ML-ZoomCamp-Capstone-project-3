import pickle

import pandas as pd
import numpy as np

import ast

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# parameters
random_seed = 42
n_estimators=500
learning_rate=0.01
max_depth=6
output_file = f'model_n_estimators={n_estimators}_max_depth={max_depth}_learning_rate={learning_rate}.bin'
path_save = '/Users/fdl/Repos/ML-ZoomCamp-Capstone-project-3/'

# load data and preparation
df = pd.read_csv(r'/Users/fdl/Repos/ML-ZoomCamp-Capstone-project-3/01 Data/cleaned_data.csv')
# rename target
df = df.rename(columns={'rating': 'y'})

# functions feature variables
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

# clean data
df['brand'] = df['model'].str.split(' ').str[0]
    
# Impute
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

df = df[feature_selection + ['y']]

# train final model
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=random_seed)

df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train.y.values
df_full_train.drop(['y'], axis=1, inplace=True)

y_test = df_test.y.values
df_test.drop(['y'], axis=1, inplace=True)

# select features based on previous analysis
num_features = [
    'price', 'storage_gb', 
    'battery_mah', 'fast_charge_w', 'screen_size_in', 
    'refresh_rate_hz', 'primary_rear_mp', 'total_pixels',
    'front_camera_mp', 'memory_card_max_gb', 'rear_camera_count'
]

cat_features = ['network_type',  'memory_card_type', 'brand']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ],
    remainder='passthrough' # Esto deja las columnas booleanas (NFC, VoLTE, etc.) como están
)

model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                    max_depth=max_depth, random_state=42)

X_full_train = df_full_train[feature_selection]
X_test = df_test[feature_selection]

results = []

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# Trainning
pipeline.fit(X_full_train, y_full_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

results.append({"Model": "XGBoost", "MAE": mae, "RMSE": rmse, "R2": r2})

# Results
import pandas as pd
results_df = pd.DataFrame(results).sort_values(by="MAE")
print(results_df)

print(f'Saving the model to {output_file}')
with open(path_save + output_file, 'wb') as f_out:
    pickle.dump(pipeline, f_out)
print(f'Model saved')