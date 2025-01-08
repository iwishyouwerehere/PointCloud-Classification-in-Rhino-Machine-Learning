from flask import Flask, request, jsonify
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

        # Get data and validate
        data = request.get_json()
        if not data or 'dataset' not in data or 'question' not in data:
            return jsonify({'error': 'Missing required data'}), 400

        # Truncate request data for logging
        truncated_dataset = [{
            'points': item['points'][:2] + ['...'] if len(item['points']) > 2 else item['points'],
            'label': item['label']
        } for item in data['dataset'][:3]] + ['...'] if len(data['dataset']) > 3 else data['dataset'][:3]

        truncated_question = {
            'points': data['question']['points'][:2] + ['...'] if len(data['question']['points']) > 2 else data['question']['points']
        }

        # Prepare response
        response_data = {'prediction': 'PREDICTION_PLACEHOLDER'}  # Placeholder

        # Log to file - Include truncated request and response
        with open('connection_logs.txt', 'a') as file:
            file.write(f"IP: {client_ip}, Date: {date_str}, Time: {time_str}, "
                       f"Request (truncated): Dataset: {json.dumps(truncated_dataset)}, Question: {json.dumps(truncated_question)}, "
                       f"Response: {json.dumps(response_data)}\n")  # Log placeholder for now

        # Log to console (for immediate debugging) - Optional
        print(f"Received request - Dataset (truncated): {json.dumps(truncated_dataset)}, Question (truncated): {json.dumps(truncated_question)}")

        # Prepare training data (using the full data, not the truncated one)
        X = [np.array(item['points']).flatten() for item in data['dataset']]
        y = [item['label'] for item in data['dataset']]

        # Train model
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)

        # Make prediction
        question_points = np.array(data['question']['points']).flatten()
        prediction = model.predict([question_points])[0]

        # Update response data with actual prediction
        response_data['prediction'] = prediction

        # Log updated response to console (for immediate debugging) - Optional
        print(f"Response: {json.dumps(response_data)}")

        return jsonify(response_data)

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Make the app remotely accessible
    app.run(debug=True, host='0.0.0.0', port=5000)
