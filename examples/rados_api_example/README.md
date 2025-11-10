# Ceph RADOS API Example

This directory contains a simple Python script demonstrating how to use the RADOS API to interact with your Ceph cluster.

## Prerequisites

1. **Install Python RADOS library:**
   ```bash
   # For Debian/Ubuntu
   sudo apt-get install python3-rados
   
   # Or using pip
   pip install rados
   ```

2. **Ceph Configuration:**
   - Ensure you have `/etc/ceph/ceph.conf` configured
   - Ensure you have admin keyring at `/etc/ceph/ceph.client.admin.keyring`

3. **Verify Ceph is accessible:**
   ```bash
   ceph -s
   ceph osd pool ls
   ```

## Configuration

Before running the script, edit `ceph_rados_example.py` and update:

- `POOL_NAME`: Set to an existing pool name (check with `ceph osd pool ls`)
- `CONF_FILE`: Path to your ceph.conf (default: `/etc/ceph/ceph.conf`)

## Running the Script

```bash
# Make it executable
chmod +x ceph_rados_example.py

# Run it
./ceph_rados_example.py

# Or with python directly
python3 ceph_rados_example.py
```

## What the Script Does

1. **Connects** to the Ceph cluster
2. **Displays** cluster statistics (space usage, objects)
3. **Lists** all available pools
4. **Creates** a test object in the specified pool
5. **Reads** the object back
6. **Shows** object statistics
7. **Lists** objects in the pool
8. **Sets** extended attributes (xattrs) on objects
9. **Appends** data to objects
10. **Cleans up** by optionally removing the test object

## Important Notes

- **CephFS vs RADOS**: While you have CephFS mounted at `/mnt/mycephfs`, the RADOS API operates at a lower level directly on the object storage. They work with the same cluster but at different abstraction levels.
- **CephFS uses metadata and data pools**: Your CephFS mount uses specific pools for metadata and data. You can use RADOS API on these pools, but be careful not to corrupt CephFS data.
- **Safe pools**: It's safer to test RADOS operations on a separate pool that's not used by CephFS.

## Creating a Test Pool

If you want a dedicated pool for testing:

```bash
# Create a new pool
ceph osd pool create testpool 32 32

# Enable the pool for use
ceph osd pool application enable testpool rgw

# Update the script's POOL_NAME to 'testpool'
```

## Troubleshooting

**Permission denied:**
```bash
# Run with sudo if you have permission issues
sudo python3 ceph_rados_example.py
```

**Module not found:**
```bash
# Install the RADOS Python bindings
sudo apt-get install python3-rados
```

**Connection issues:**
```bash
# Check Ceph status
ceph -s

# Verify config file exists
cat /etc/ceph/ceph.conf
```
