# vibe coded from Gemini, needs fixing
import routeros_api
import os


class MikroTikManager:
    def __init__(self, host, user, password):
        self.pool = routeros_api.RouterOsApiPool(
            host, 
            username=user, 
            password=password, 
            plaintext_login=True,
            port=8729,
            ssl_verify=False,
            use_ssl=True
        )
        self.api = None

    def __enter__(self):
        self.api = self.pool.get_api()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.disconnect()
    
    def add_vlan(self, name, vlan_id, interface, comment=None):
        """Creates a VLAN interface on top of a physical or bridge interface."""
        vlan_resource = self.api.get_resource('/interface/vlan')
        params = {
            'name': name,
            'vlan-id': str(vlan_id),
            'interface': interface
        }
        if comment: params['comment'] = comment
        
        vlan_resource.add(**params)
        print(f"VLAN '{name}' (ID: {vlan_id}) created on {interface}.")
    def add_vrf(self, name, interfaces, comment=None):
        """
        Creates a new VRF instance.
        :param name: The name of the VRF.
        :param interfaces: A list of interface names (e.g., ['ether2', 'vlan10']).
        :param comment: Optional description.
        """
        vrf_resource = self.api.get_resource('/ip/vrf')
        
        # Interfaces in the API must be a comma-separated string
        if isinstance(interfaces, list):
            interfaces = ",".join(interfaces)

        params = {
            'name': name,
            'interfaces': interfaces
        }
        
        if comment:
            params['comment'] = comment

        try:
            vrf_resource.add(**params)
            print(f"VRF '{name}' created successfully with interfaces: {interfaces}")
        except Exception as e:
            print(f"Failed to create VRF: {e}")

    def add_firewall_rule(self, chain, action, protocol=None, dst_port=None, comment=None, **extra_params):
        """
        Creates a new IPv4 firewall filter rule.
        """
        firewall = self.api.get_resource('/ip/firewall/filter')
        
        # Construct the arguments dictionary
        params = {
            'chain': chain,
            'action': action,
        }
        
        # Add optional parameters if provided
        if protocol:
            params['protocol'] = protocol
        if dst_port:
            params['dst-port'] = str(dst_port)
        if comment:
            params['comment'] = comment
        params.update(extra_params)
        try:
            firewall.add(**params)
            print(f"Successfully added {action} rule to {chain} chain.")
        except Exception as e:
            print(f"Failed to add firewall rule: {e}")
    
    def add_ip_address(self, address, interface, network=None, comment=None):
        """Assigns an IP address to a specific interface."""
        address_resource = self.api.get_resource('/ip/address')
        params = {
            'address': address, # Format: '192.168.10.1/24'
            'interface': interface
        }
        if network: params['network'] = network
        if comment: params['comment'] = comment
        
        address_resource.add(**params)
        print(f"Assigned {address} to {interface}.")

    def modify_bridge(self, name, vlan_filtering=None, comment=None, frame_types=None, **extra_args):
        """
        Modifies an existing bridge interface. 
        Useful for enabling 'vlan-filtering' which is disabled by default.
        """
        bridge_resource = self.api.get_resource('/interface/bridge')
        
        # Find the internal ID of the bridge by its name
        bridge_data = bridge_resource.get(name=name)
        if not bridge_data:
            print(f"Error: Bridge '{name}' not found.")
            return

        bridge_id = bridge_data[0]['id']
        params = {'id': bridge_id}
        
        # Update specific properties
        if vlan_filtering is not None:
            params['vlan-filtering'] = 'yes' if vlan_filtering else 'no'
        if frame_types:
            params['frame-types'] = frame_types # e.g., 'admit-only-vlan-tagged'
        if comment:
            params['comment'] = comment
        params.update(extra_args)
        bridge_resource.set(**params)
        print(f"Bridge '{name}' updated successfully.")

# Usage Example
if __name__ == "__main__":
    # Using Safe Mode = True for network-sensitive changes
    # TODO: Add Safe Mode API call.
    with MikroTikManager(os.environ["MIKROTIK_ENDPOINT"], os.environ["MIKROTIK_USER"], os.environ["MIKROTIK_PASSWORD"]) as mt:
        # 1. Create a VRF named 'Customer_A'
        # 2. Assign ether2 and ether3 to it
        vlan_name = "newTestVLAN"
        # mt.add_vlan(vlan_name, 1234, "testBridge", "Testing MikroTik automation")
        mt.add_vrf(
            name='Customer_A', 
            interfaces=[vlan_name],
            comment='Isolated routing for Customer A'
        )


