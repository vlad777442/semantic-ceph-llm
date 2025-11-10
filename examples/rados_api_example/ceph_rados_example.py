#!/usr/bin/env python3
"""
Simple Ceph RADOS API Example
This script demonstrates basic operations with the RADOS API on a Ceph cluster.
"""

import rados
import sys

def main():
    # Configuration - adjust these values for your setup
    CLUSTER_NAME = 'ceph'
    CONF_FILE = '/etc/ceph/ceph.conf'
    KEYRING_FILE = '/etc/ceph/ceph.client.admin.keyring'
    POOL_NAME = 'cephfs.cephfs.data'  # Change to your pool name (use: sudo ceph osd pool ls)
    
    print("=" * 50)
    print("Ceph RADOS API Example")
    print("=" * 50)
    print("\nNote: This script requires sudo access to read Ceph keyring")
    print("Run with: sudo ./ceph_rados_example.py\n")
    
    try:
        # 1. Connect to the Ceph cluster
        print("\n1. Connecting to Ceph cluster...")
        cluster = rados.Rados(
            conffile=CONF_FILE,
            name='client.admin',
            clustername=CLUSTER_NAME
        )
        # Read the keyring file
        cluster.conf_read_file(CONF_FILE)
        cluster.connect()
        print(f"   ✓ Connected to cluster: {cluster.get_fsid()}")
        
        # 2. Get cluster statistics
        print("\n2. Cluster Statistics:")
        stats = cluster.get_cluster_stats()
        print(f"   Total space: {stats['kb'] / (1024**2):.2f} GB")
        print(f"   Used space: {stats['kb_used'] / (1024**2):.2f} GB")
        print(f"   Available space: {stats['kb_avail'] / (1024**2):.2f} GB")
        print(f"   Number of objects: {stats['num_objects']}")
        
        # 3. List all pools
        print("\n3. Available Pools:")
        pools = cluster.list_pools()
        for pool in pools:
            print(f"   - {pool}")
        
        # 4. Work with a specific pool
        print(f"\n4. Working with pool: {POOL_NAME}")
        
        # Check if pool exists, if not, list available pools
        if POOL_NAME not in pools:
            print(f"   ⚠ Pool '{POOL_NAME}' not found!")
            print(f"   Please change POOL_NAME to one of: {', '.join(pools)}")
            cluster.shutdown()
            return
        
        # Open IO context for the pool
        ioctx = cluster.open_ioctx(POOL_NAME)
        
        # 5. Write an object to the pool
        print("\n5. Writing test object...")
        object_name = "test_object"
        test_data = b"Hello from RADOS API! This is test data."
        ioctx.write_full(object_name, test_data)
        print(f"   ✓ Written object '{object_name}' ({len(test_data)} bytes)")
        
        # 6. Read the object back
        print("\n6. Reading test object...")
        read_data = ioctx.read(object_name)
        print(f"   ✓ Read {len(read_data)} bytes")
        print(f"   Data: {read_data.decode('utf-8')}")
        
        # 7. Get object stats
        print("\n7. Object Statistics:")
        stat = ioctx.stat(object_name)
        print(f"   Size: {stat[0]} bytes")
        print(f"   Modification time: {stat[1]}")
        
        # 8. List objects in the pool
        print("\n8. Listing objects in pool:")
        object_iterator = ioctx.list_objects()
        count = 0
        for obj in object_iterator:
            print(f"   - {obj.key}")
            count += 1
            if count >= 10:  # Limit to first 10 objects
                print(f"   ... (showing first 10 objects)")
                break
        
        # 9. Write object with extended attributes (xattrs)
        print("\n9. Working with extended attributes...")
        xattr_name = "user.description"
        xattr_value = b"This is a test object created by RADOS API"
        ioctx.set_xattr(object_name, xattr_name, xattr_value)
        print(f"   ✓ Set xattr '{xattr_name}'")
        
        # Read xattr back
        retrieved_xattr = ioctx.get_xattr(object_name, xattr_name)
        print(f"   ✓ Retrieved xattr: {retrieved_xattr.decode('utf-8')}")
        
        # 10. Append to object
        print("\n10. Appending to object...")
        append_data = b" Additional data appended!"
        ioctx.append(object_name, append_data)
        print(f"   ✓ Appended {len(append_data)} bytes")
        
        # Read the complete object
        complete_data = ioctx.read(object_name)
        print(f"   Complete data: {complete_data.decode('utf-8')}")
        
        # 11. Remove the test object (cleanup)
        print("\n11. Cleanup...")
        response = input("   Remove test object? (y/n): ")
        if response.lower() == 'y':
            ioctx.remove_object(object_name)
            print(f"   ✓ Removed object '{object_name}'")
        else:
            print(f"   ✓ Keeping object '{object_name}'")
        
        # Close the IO context
        ioctx.close()
        
        # 12. Disconnect from cluster
        print("\n12. Disconnecting from cluster...")
        cluster.shutdown()
        print("   ✓ Disconnected")
        
        print("\n" + "=" * 50)
        print("Example completed successfully!")
        print("=" * 50)
        
    except rados.Error as e:
        print(f"\n❌ RADOS Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
