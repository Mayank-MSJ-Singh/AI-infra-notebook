from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Simulate ML prediction
    input_data = np.array(data['features'])
    prediction = float(np.sum(input_data))  # Dummy model

    return jsonify({
    'prediction': prediction,
    'model_version': 'v2.0'  # Changed
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)