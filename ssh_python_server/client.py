import requests
import time

SERVER_URL = "https://localhost:5001/keys"
LOCAL_KEY_STORE = "fetched_keys.txt"
POLL_INTERVAL = 30

def fetch_keys():
    try:
        response = requests.get(SERVER_URL)
        if response.status_code == 200:
            return set(response.json().get('keys', []))
    except Exception as e:
        print(f"Error connecting to server: {e}")
    return set()

def update_local_keys(new_keys):
    # Read existing keys to avoid duplicates
    existing_keys = set()
    if os.path.exists(LOCAL_KEY_STORE):
        with open(LOCAL_KEY_STORE, 'r') as f:
            existing_keys = set(line.strip() for line in f)

    # Find keys we don't have yet
    keys_to_add = new_keys - existing_keys

    if keys_to_add:
        with open(LOCAL_KEY_STORE, 'a') as f:
            for key in keys_to_add:
                f.write(key + "\n")
        print(f"Added {len(keys_to_add)} new keys.")
    else:
        print("No new keys found.")

if __name__ == "__main__":
    import os
    print(f"Starting client. Polling {SERVER_URL} every {POLL_INTERVAL}s...")
    while True:
        remote_keys = fetch_keys()
        if remote_keys:
            update_local_keys(remote_keys)
        time.sleep(POLL_INTERVAL)