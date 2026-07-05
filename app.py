from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import joblib
import os
import traceback
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Paths to model and scaler files for liquid and gas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    'liquid': os.path.join(BASE_DIR, 'liq', 'reynolds_model_liq.keras'),
    'gas': os.path.join(BASE_DIR, 'gas', 'reynolds_model_gas.keras')
}
FEATURE_SCALER_PATHS = {
    'liquid': os.path.join(BASE_DIR, 'liq', 'feature_scaler_liq.pkl'),
    'gas': os.path.join(BASE_DIR, 'gas', 'feature_scaler_gas.pkl')
}
TARGET_SCALER_PATHS = {
    'liquid': os.path.join(BASE_DIR, 'liq', 'target_scaler_liq.pkl'),
    'gas': os.path.join(BASE_DIR, 'gas', 'target_scaler_gas.pkl')
}

# Check if files exist
for medium in ['liquid', 'gas']:
    for path in [MODEL_PATHS[medium], FEATURE_SCALER_PATHS[medium], TARGET_SCALER_PATHS[medium]]:
        if not os.path.exists(path):
            print(f"File not found: {path}")
        else:
            print(f"File found: {path}")

# Load models and scalers
models = {}
feature_scalers = {}
target_scalers = {}
for medium in ['liquid', 'gas']:
    try:
        models[medium] = tf.keras.models.load_model(MODEL_PATHS[medium])
        feature_scalers[medium] = joblib.load(FEATURE_SCALER_PATHS[medium])
        target_scalers[medium] = joblib.load(TARGET_SCALER_PATHS[medium])
        print(f"{medium.capitalize()} model and scalers loaded successfully")
        print(f"{medium.capitalize()} feature scaler type: MinMaxScaler")
        print(f"{medium.capitalize()} feature scaler data_min:", feature_scalers[medium].data_min_)
        print(f"{medium.capitalize()} feature scaler data_max:", feature_scalers[medium].data_max_)
        print(f"{medium.capitalize()} target scaler data_min:", target_scalers[medium].data_min_)
        print(f"{medium.capitalize()} target scaler data_max:", target_scalers[medium].data_max_)
    except FileNotFoundError as e:
        print(f"FileNotFoundError for {medium}: {str(e)}")
        print(f"Ensure {MODEL_PATHS[medium]}, {FEATURE_SCALER_PATHS[medium]}, and {TARGET_SCALER_PATHS[medium]} exist in {BASE_DIR}")
    except Exception as e:
        print(f"Unexpected error loading {medium} model or scalers: {str(e)}")
        print("Stack trace:", traceback.format_exc())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    medium = data.get('medium', 'liquid')
    
    if medium not in models or models[medium] is None:
        return jsonify({'error': f'{medium.capitalize()} model not loaded'}), 500

    try:
        density = float(data['density'])
        velocity = float(data['velocity'])
        diameter = float(data['length'])  # Matches training 'Diameter'
        viscosity = float(data['viscosity'])
        
        # Validate inputs
        if density <= 0 or velocity <= 0 or diameter <= 0 or viscosity <= 0:
            print(f"Invalid input: Non-positive values: density={density}, velocity={velocity}, diameter={diameter}, viscosity={viscosity}")
            return jsonify({'error': 'All input values must be positive'}), 400
        if medium not in ['liquid', 'gas']:
            print(f"Invalid medium: {medium}")
            return jsonify({'error': 'Invalid medium specified'}), 400
        
        # Validate ranges based on training data
        if medium == 'liquid':
            if not (715.6277903038863 <= density <= 1032.9733671921813):
                print(f"Invalid liquid density: {density}")
                return jsonify({'error': 'Density must be between 715.6277903038863 and 1032.9733671921813 for liquid'}), 400
            if not (0 <= velocity <= 8.451995705516785):
                print(f"Invalid liquid velocity: {velocity}")
                return jsonify({'error': 'Velocity must be between 0 and 8.451995705516785 for liquid'}), 400
            if not (0 <= diameter <= 0.36748163999068206):
                print(f"Invalid liquid diameter: {diameter}")
                return jsonify({'error': 'Diameter must be between 0 and 0.36748163999068206 for liquid'}), 400
            if not (7.699317799509498e-05 <= viscosity <= 0.006080380547145805):
                print(f"Invalid liquid viscosity: {viscosity}")
                return jsonify({'error': 'Viscosity must be between 7.699317799509498e-05 and 0.006080380547145805 for liquid'}), 400
        else:  # gas
            if not (0 <= density <= 20.36578206659651):
                print(f"Invalid gas density: {density}")
                return jsonify({'error': 'Density must be between 0 and 20.36578206659651 for gas'}), 400
            if not (0 <= velocity <= 52.499443580940465):
                print(f"Invalid gas velocity: {velocity}")
                return jsonify({'error': 'Velocity must be between 0 and 52.499443580940465 for gas'}), 400
            if not (0 <= diameter <= 0.5249885104741552):
                print(f"Invalid gas diameter: {diameter}")
                return jsonify({'error': 'Diameter must be between 0 and 0.5249885104741552 for gas'}), 400
            if not (6.100923200764568e-07 <= viscosity <= 2.8885055939928103e-05):
                print(f"Invalid gas viscosity: {viscosity}")
                return jsonify({'error': 'Viscosity must be between 6.100923200764568e-07 and 2.8885055939928103e-05 for gas'}), 400
        
        print(f"Received inputs: density={density}, velocity={velocity}, diameter={diameter}, viscosity={viscosity}, medium={medium}")
        
        # Prepare input
        input_data = np.array([[density, velocity, diameter, viscosity]])
        input_log = np.log1p(input_data)
        print(f"Log-transformed input: {input_log}")
        
        # Scale input
        input_scaled = feature_scalers[medium].transform(input_log)
        print(f"Scaled input: {input_scaled}")
        
        # Predict
        pred_scaled = models[medium].predict(input_scaled, verbose=0)
        print(f"Raw model prediction (scaled): {pred_scaled}")
        pred_log = target_scalers[medium].inverse_transform(pred_scaled)
        print(f"Inverse-transformed prediction (log): {pred_log}")
        predicted_re = float(np.expm1(pred_log)[0][0])
        print(f"Raw predicted_re: {predicted_re}")
        
        # Handle invalid predictions
        if np.isnan(predicted_re) or predicted_re < 0:
            print(f"Invalid predicted_re: {predicted_re}, setting to 1.0")
            predicted_re = 1.0
        
        # Calculate actual Re
        actual_re = float((density * velocity * diameter) / viscosity)
        
        # Calculate MAPE
        mape = float(100 * abs(actual_re - predicted_re) / (actual_re + 1e-10))
        
        print(f"Final result: actual_re={actual_re}, predicted_re={predicted_re}, mape={mape}")
        
        return jsonify({
            'actual_re': actual_re,
            'predicted_re': predicted_re,
            'mape': mape,
            'density': density,
            'velocity': velocity,
            'length': diameter,  # Return as length for UI
            'viscosity': viscosity,
            'medium': medium
        })
    except KeyError as e:
        print(f"KeyError: Missing parameter: {str(e)}")
        return jsonify({'error': f'Missing parameter: {str(e)}'}), 400
    except ValueError as e:
        print(f"ValueError: Invalid input: {str(e)}")
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)