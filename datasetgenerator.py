from flask import Flask, request, jsonify
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import datetime  # Import datetime module

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
        
        # Write to text file
        with open('connection_logs.txt', 'a') as file:
            file.write(f"IP: {client_ip}, Date: {date_str}, Time: {time_str}\n")
        
        # Print received data for debugging
        print("Received request data:", request.get_json())
        
        # Get both dataset and question from the request
        data = request.get_json()
        if not data or 'dataset' not in data or 'question' not in data:
            return jsonify({'error': 'Missing required data'}), 400
            
        dataset = data['dataset']
        question = data['question']
        
        # Prepare training data
        X = []  # points
        y = []  # labels
        
        for item in dataset:
            X.append(np.array(item['points']).flatten())
            y.append(item['label'])
        
        # Train model
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)
        
        # Make prediction
        question_points = np.array(question['points']).flatten()
        prediction = model.predict([question_points])[0]
        
        return jsonify({'prediction': prediction})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
