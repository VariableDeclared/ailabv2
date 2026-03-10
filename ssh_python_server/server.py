from flask import Flask, request, jsonify, abort
import json
import os

app = Flask(__name__)
DB_FILE = "keys.json"
# In a real app, use: os.environ.get("API_SECRET")
API_SECRET = "super-secret-token-123"

def get_keys():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, 'r') as f:
        return json.load(f)

@app.route('/keys', methods=['GET'])
def list_keys():
    """Public endpoint: Anyone can see the keys to sync them."""
    return jsonify({"keys": get_keys()})

@app.route('/keys', methods=['POST'])
def add_key():
    """Protected endpoint: Requires X-API-Key header."""
    provided_key = request.headers.get("X-API-Key")
    
    if provided_key != API_SECRET:
        return jsonify({"error": "Unauthorized"}), 401

    new_ssh_key = request.json.get('key')
    if not new_ssh_key:
        return jsonify({"error": "No key provided"}), 400
    
    keys = get_keys()
    if new_ssh_key not in keys:
        keys.append(new_ssh_key)
        with open(DB_FILE, 'w') as f:
            json.dump(keys, f)
            
    return jsonify({"message": "Key added successfully"}), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)