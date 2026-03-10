# SSH Key Sync: Simple API & Polling Client
This project provides a lightweight, automated way to manage and distribute public SSH keys across a fleet of servers. It consists of a Centralized Flask API (the "Source of Truth") and a Python Polling Client that runs on target machines to keep their local key stores up to date.

## Architecture Overview
The system operates on a simple pull-based model:

Server: Acts as a repository for public keys, protected by an API Secret for write access.

Client: Regularly fetches the list of keys and appends any new, unrecognized keys to a local file.

## Features
API Security: Uses X-API-Key header authentication to prevent unauthorized key injection.

Idempotent Updates: The client checks for existing keys before appending, preventing duplicate entries in your key file.

Lightweight: Minimal dependencies (Flask and Requests).

Automatic Sync: Configurable polling interval (default: 30 seconds).

## Installation
Clone the repository:

Bash
git clone https://github.com/your-username/ssh-key-sync.git
cd ssh-key-sync
Install requirements:

Bash
pip install flask requests
## Components
1. The Server (server.py)
The server hosts the API. To run it:

Bash
python server.py
Public Endpoint: GET /keys — Returns a JSON list of all public keys.

Protected Endpoint: POST /keys — Adds a new key. Requires the X-API-Key header.

2. The Client (client.py)
The client runs on any machine where you want keys to be automatically updated.

Bash
python client.py
By default, it polls the server every 30 seconds and saves keys to fetched_keys.txt.

## API Usage Guide
Adding a Key (Authenticated)
To add a new SSH key to the server's database, send a POST request with your secret:

Bash
curl -X POST http://<server-ip>:5000/keys \
     -H "X-API-Key: super-secret-token-123" \
     -H "Content-Type: application/json" \
     -d '{"key": "ssh-rsa AAAAB3Nza...user@host"}'
Fetching Keys (Public)
To see the current list of synchronized keys:

Bash
curl http://<server-ip>:5000/keys
## Configuration
Variable	Description	Location
API_SECRET	The token required to POST new keys.	server.py
POLL_INTERVAL	Seconds between client checks.	client.py
LOCAL_KEY_STORE	File where the client saves keys.	client.py
## Security Recommendations
Production SSL: In a production environment, wrap the Flask app in a production server like Gunicorn and use Nginx to provide HTTPS.

Environment Variables: Move the API_SECRET out of the source code and into an .env file or system environment variable.

Authorized Keys: To use this for real SSH access, point the client's LOCAL_KEY_STORE to ~/.ssh/authorized_keys (ensure proper file permissions).
