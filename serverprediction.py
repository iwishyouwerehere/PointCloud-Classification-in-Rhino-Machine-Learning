from flask import Flask, request, jsonify, send_file
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import datetime
import json

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get client IP address
        client_ip = request.remote_addr

        # Get current date and time
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        # Get and validate request data
        request_data = request.get_json()
        if not request_data or 'dataset' not in request_data or 'question' not in request_data:
            return jsonify({'error': 'Missing required data'}), 400

        # Prepare training data
        X = [np.array(item['points']).flatten() for item in request_data['dataset']]
        y = [item['label'] for item in request_data['dataset']]

        # Train model
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)

        # Make prediction
        question_points = np.array(request_data['question']['points']).flatten()
        prediction = model.predict([question_points])[0]

        # Prepare response
        response_data = {'prediction': prediction}

        # Log to file - Include full request and response
        with open('connection_logs.txt', 'a') as file:
            file.write(f"IP: {client_ip}, Date: {date_str}, Time: {time_str}, "
                       f"Request: {json.dumps(request_data)}, "
                       f"Response: {json.dumps(response_data)}\n")

        # Log to console (for immediate debugging) - Optional
        print(f"Received request: {json.dumps(request_data)}")
        print(f"Response: {json.dumps(response_data)}")

        return jsonify(response_data)

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 400

@app.route('/download_logs')
def download_logs():
    try:
        return send_file('connection_logs.txt', as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Make the app remotely accessible
    app.run(debug=True, host='0.0.0.0', port=5000)
