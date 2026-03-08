# vibe coded from Gemini, needs fixing
import routeros_api
import os

def main():
    connection = routeros_api.RouterOsApiPool(
        '192.168.254.1',
        username=os.environ["MIKROTIK_USER"],
        password=os.environ["MIKROTIK_PASSWORD"],
        plaintext_login=True, # Required for newer RouterOS versions via API
        use_ssl=True,
        ssl_verify=False,
        port=8729
    )

    try:
        # Establish the connection
        api = connection.get_api()

        # 1. Get System Identity
        # Equivalent to /system identity print
        identity_resource = api.get_resource('/system/identity')
        identity = identity_resource.get()
        print(f"Device Identity: {identity[0]['name']}")
        print("-" * 30)

        # 2. Get All Interfaces
        # Equivalent to /interface print
        interface_resource = api.get_resource('/interface')
        interfaces = interface_resource.get()

        print(f"{'Name':<20} {'Type':<15} {'Enabled':<10}")
        for iface in interfaces:
            name = iface.get('name')
            type_ = iface.get('type')
            disabled = iface.get('disabled')
            
            # Convert 'false' (disabled) to 'Yes' (enabled) for readability
            status = "Yes" if disabled == 'false' else "No"
            print(f"{name:<20} {type_:<15} {status:<10}")

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Always disconnect to free up the API slot
        connection.disconnect()

if __name__ == "__main__":
    main()