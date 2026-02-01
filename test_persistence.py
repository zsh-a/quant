#!/usr/bin/env python3
"""
State persistence and recovery test script.
Tests checkpoint creation, loading, and session restoration.
"""

import sys
import os
import requests
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

API_BASE = "http://localhost:8000"


def test_create_checkpoint(session_id: str):
    """Test creating a checkpoint"""
    print(f"\n{'='*60}")
    print(f"TEST: Create Checkpoint")
    print(f"{'='*60}")
    
    try:
        response = requests.post(f"{API_BASE}/session/{session_id}/checkpoint")
        response.raise_for_status()
        
        data = response.json()
        print(f"✓ Checkpoint created: {data}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to create checkpoint: {e}")
        return False


def test_list_checkpoints(session_id: str):
    """Test listing checkpoints"""
    print(f"\n{'='*60}")
    print(f"TEST: List Checkpoints")
    print(f"{'='*60}")
    
    try:
        response = requests.get(f"{API_BASE}/session/{session_id}/checkpoints")
        response.raise_for_status()
        
        data = response.json()
        checkpoints = data.get('checkpoints', [])
        
        print(f"Found {len(checkpoints)} checkpoint(s):")
        for i, cp in enumerate(checkpoints, 1):
            print(f"  {i}. Time: {cp['checkpoint_time']}")
            print(f"     Size: {cp['size_bytes']} bytes ({cp['size_bytes']/1024:.2f} KB)")
            if cp.get('metadata'):
                print(f"     Metadata: {cp['metadata']}")
        
        print(f"✓ Listed {len(checkpoints)} checkpoints")
        return True
        
    except Exception as e:
        print(f"✗ Failed to list checkpoints: {e}")
        return False


def test_restore_session(session_id: str):
    """Test restoring a session"""
    print(f"\n{'='*60}")
    print(f"TEST: Restore Session")
    print(f"{'='*60}")
    
    try:
        response = requests.post(f"{API_BASE}/session/{session_id}/restore")
        response.raise_for_status()
        
        data = response.json()
        print(f"✓ Session restored:")
        print(f"  Checkpoint time: {data.get('checkpoint_time')}")
        print(f"  Status: {data.get('status')}")
        print(f"  Progress: {data.get('progress', 0):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to restore session: {e}")
        return False


def test_persistence_stats():
    """Test getting persistence statistics"""
    print(f"\n{'='*60}")
    print(f"TEST: Persistence Statistics")
    print(f"{'='*60}")
    
    try:
        response = requests.get(f"{API_BASE}/persistence/stats")
        response.raise_for_status()
        
        stats = response.json()
        print(f"Persistence Statistics:")
        print(f"  Sessions: {stats.get('session_count', 0)}")
        print(f"  Checkpoints: {stats.get('checkpoint_count', 0)}")
        print(f"  Total size: {stats.get('total_size_mb', 0):.2f} MB")
        
        print(f"✓ Statistics retrieved")
        return True
        
    except Exception as e:
        print(f"✗ Failed to get statistics: {e}")
        return False


def test_checkpoint_compression():
    """Test checkpoint compression"""
    print(f"\n{'='*60}")
    print(f"TEST: Checkpoint Compression")
    print(f"{'='*60}")
    
    from src.api.state_persistence import persistence
    
    # Create a large state
    large_state = {
        'data': ['x' * 1000 for _ in range(100)],  # 100KB of data
        'numbers': list(range(10000))
    }
    
    # Serialize with compression
    compressed = persistence.serialize_state(large_state)
    
    # Serialize without compression
    persistence.use_compression = False
    uncompressed = persistence.serialize_state(large_state)
    persistence.use_compression = True
    
    compression_ratio = len(uncompressed) / len(compressed)
    
    print(f"Uncompressed size: {len(uncompressed)} bytes ({len(uncompressed)/1024:.2f} KB)")
    print(f"Compressed size: {len(compressed)} bytes ({len(compressed)/1024:.2f} KB)")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Verify deserialization
    restored = persistence.deserialize_state(compressed)
    assert restored == large_state, "Deserialization failed"
    
    print(f"✓ Compression working correctly ({compression_ratio:.2f}x reduction)")
    return True


def test_recovery_time(session_id: str):
    """Test recovery time"""
    print(f"\n{'='*60}")
    print(f"TEST: Recovery Time")
    print(f"{'='*60}")
    
    import time
    
    try:
        start_time = time.time()
        response = requests.post(f"{API_BASE}/session/{session_id}/restore")
        response.raise_for_status()
        recovery_time = time.time() - start_time
        
        print(f"Recovery time: {recovery_time:.3f}s")
        
        if recovery_time < 5.0:
            print(f"✓ Recovery time within target (<5s)")
            return True
        else:
            print(f"⚠ Recovery time exceeds target (>5s)")
            return False
            
    except Exception as e:
        print(f"✗ Failed to test recovery time: {e}")
        return False


def main():
    """Run all persistence tests"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_persistence.py <session_id>")
        print("\nExample:")
        print("  python test_persistence.py abc123")
        return 1
    
    session_id = sys.argv[1]
    
    print("\n" + "#"*60)
    print("# State Persistence Test Suite")
    print("#"*60)
    print(f"Session ID: {session_id}")
    
    results = {}
    
    # Test 1: Create checkpoint
    results['create_checkpoint'] = test_create_checkpoint(session_id)
    
    # Test 2: List checkpoints
    results['list_checkpoints'] = test_list_checkpoints(session_id)
    
    # Test 3: Restore session
    results['restore_session'] = test_restore_session(session_id)
    
    # Test 4: Persistence stats
    results['persistence_stats'] = test_persistence_stats()
    
    # Test 5: Compression
    results['compression'] = test_checkpoint_compression()
    
    # Test 6: Recovery time
    results['recovery_time'] = test_recovery_time(session_id)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())
