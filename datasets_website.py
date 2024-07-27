from flask import Flask, request, send_file, jsonify
from werkzeug.exceptions import Unauthorized
import secrets
import time
import os

app = Flask(__name__)

# In-memory storage for API keys (in a real-world scenario, use a database)
api_keys = {}

# Secret key for API key generation (in a real-world scenario, use a more secure method)
SECRET_KEY = "your_secret_key_here"

# API key expiration time (in seconds)
API_KEY_EXPIRATION = 3600  # 1 hour

@app.route('/issue_api_key', methods=['POST'])
def issue_api_key():
    # Generate a new API key
    api_key = secrets.token_urlsafe(32)
    
    # Set expiration time
    expiration_time = int(time.time()) + API_KEY_EXPIRATION
    
    # Store the API key with its expiration time
    api_keys[api_key] = expiration_time
    
    return jsonify({'api_key': api_key, 'expires_at': expiration_time})

@app.route('/validate_api_key', methods=['GET'])
def validate_api_key():
    api_key = request.headers.get('X-API-Key')
    
    if not api_key:
        raise Unauthorized("API key is missing")
    
    if api_key not in api_keys:
        raise Unauthorized("Invalid API key")
    
    expiration_time = api_keys[api_key]
    current_time = int(time.time())
    
    if current_time > expiration_time:
        del api_keys[api_key]
        raise Unauthorized("API key has expired")
    
    return jsonify({'valid': True, 'expires_at': expiration_time})

@app.route('/mydataset.npz', methods=['GET'])
def get_dataset():
    api_key = request.headers.get('X-API-Key')
    
    if not api_key:
        raise Unauthorized("API key is missing")
    
    if api_key not in api_keys:
        raise Unauthorized("Invalid API key")
    
    expiration_time = api_keys[api_key]
    current_time = int(time.time())
    
    if current_time > expiration_time:
        del api_keys[api_key]
        raise Unauthorized("API key has expired")
    
    # Assuming the dataset file is in the same directory as the script
    file_path = os.path.join(os.path.dirname(__file__), 'mydataset.npz')
    
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)