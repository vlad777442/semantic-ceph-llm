#!/usr/bin/env python3
"""
Create test documents in RADOS for testing semantic search.
"""

import rados

# Test documents
test_docs = {
    "doc1_ml.txt": """
Machine Learning Algorithms Overview

Machine learning is a subset of artificial intelligence that focuses on
building systems that can learn from data. Common algorithms include:

1. Supervised Learning:
   - Linear Regression
   - Logistic Regression
   - Support Vector Machines
   - Decision Trees
   - Random Forests
   - Neural Networks

2. Unsupervised Learning:
   - K-Means Clustering
   - Hierarchical Clustering
   - Principal Component Analysis
   - Association Rules

3. Deep Learning:
   - Convolutional Neural Networks (CNN)
   - Recurrent Neural Networks (RNN)
   - Transformers
   - Generative Adversarial Networks (GAN)
""",
    
    "doc2_db.txt": """
Database Design Best Practices

When designing a database schema, consider:

1. Normalization:
   - First Normal Form (1NF)
   - Second Normal Form (2NF)
   - Third Normal Form (3NF)
   - Boyce-Codd Normal Form (BCNF)

2. Indexing:
   - Primary keys
   - Foreign keys
   - Composite indexes
   - Covering indexes

3. Performance:
   - Query optimization
   - Connection pooling
   - Caching strategies
   - Partitioning

4. Security:
   - Access control
   - Encryption at rest
   - SQL injection prevention
   - Audit logging
""",
    
    "doc3_web.txt": """
Web Development Technologies

Modern web development involves multiple technologies:

Frontend:
- HTML5 for structure
- CSS3 for styling
- JavaScript frameworks: React, Vue, Angular
- TypeScript for type safety
- Webpack for bundling

Backend:
- Node.js with Express
- Python with Django/Flask
- Java with Spring Boot
- Go for microservices
- Ruby on Rails

Databases:
- PostgreSQL
- MongoDB
- Redis for caching
- Elasticsearch for search

DevOps:
- Docker containers
- Kubernetes orchestration
- CI/CD pipelines
- Cloud platforms: AWS, Azure, GCP
""",
    
    "doc4_algo.txt": """
Algorithm Complexity and Data Structures

Understanding time and space complexity:

Big-O Notation:
- O(1): Constant time
- O(log n): Logarithmic time
- O(n): Linear time
- O(n log n): Linearithmic time
- O(n²): Quadratic time
- O(2ⁿ): Exponential time

Common Data Structures:
1. Arrays: O(1) access, O(n) search
2. Linked Lists: O(1) insertion, O(n) search
3. Hash Tables: O(1) average case operations
4. Binary Trees: O(log n) operations
5. Heaps: O(log n) insertion/deletion
6. Graphs: Various traversal algorithms
""",
    
    "doc5_ceph.txt": """
Ceph Storage Architecture

Ceph is a distributed storage system providing:

Components:
1. RADOS: Reliable Autonomic Distributed Object Store
   - OSDs (Object Storage Daemons)
   - Monitors for cluster state
   - CRUSH algorithm for data placement

2. Storage Types:
   - Object Storage (RADOS Gateway)
   - Block Storage (RBD)
   - File System (CephFS)

3. Features:
   - Self-healing
   - Self-managing
   - No single point of failure
   - Scalability to exabytes

4. Use Cases:
   - Cloud infrastructure
   - Big data analytics
   - Media storage
   - Backup solutions
"""
}

def main():
    print("Creating test documents in RADOS...")
    
    # Connect to Ceph
    cluster = rados.Rados(conffile='/etc/ceph/ceph.conf', name='client.admin')
    cluster.conf_read_file('/etc/ceph/ceph.conf')
    cluster.connect()
    
    # Open pool
    ioctx = cluster.open_ioctx('cephfs.cephfs.data')
    
    # Write test documents
    for filename, content in test_docs.items():
        try:
            ioctx.write_full(filename, content.encode('utf-8'))
            print(f"✅ Created: {filename} ({len(content)} bytes)")
        except Exception as e:
            print(f"❌ Failed to create {filename}: {e}")
    
    # Close connections
    ioctx.close()
    cluster.shutdown()
    
    print(f"\n✅ Created {len(test_docs)} test documents")
    print("\nYou can now test semantic search with queries like:")
    print("  - 'machine learning neural networks'")
    print("  - 'database indexing and optimization'")
    print("  - 'web development frameworks'")
    print("  - 'algorithm complexity'")
    print("  - 'distributed storage systems'")

if __name__ == "__main__":
    main()
