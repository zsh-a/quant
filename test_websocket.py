#!/usr/bin/env python3
"""
WebSocket client test script.
Tests WebSocket connection, message handling, and real-time updates.
"""

import asyncio
import websockets
import json
from datetime import datetime

WS_URL = "ws://localhost:8000/ws"

async def test_websocket_connection(session_id: str):
    """Test WebSocket connection and message handling"""
    print(f"\n{'='*60}")
    print(f"Testing WebSocket Connection")
    print(f"{'='*60}")
    print(f"Session ID: {session_id}")
    print(f"Connecting to: {WS_URL}/{session_id}")
    
    try:
        async with websockets.connect(f"{WS_URL}/{session_id}") as websocket:
            print("✓ WebSocket connected successfully")
            
            # Send subscribe message
            subscribe_msg = {
                "type": "subscribe",
                "session_id": session_id
            }
            await websocket.send(json.dumps(subscribe_msg))
            print(f"→ Sent subscribe message")
            
            # Listen for messages
            message_count = 0
            start_time = datetime.now()
            
            print("\nListening for messages (Ctrl+C to stop)...")
            print(f"{'='*60}\n")
            
            async for message in websocket:
                message_count += 1
                data = json.loads(message)
                
                msg_type = data.get('type')
                timestamp = data.get('timestamp', 'N/A')
                
                if msg_type == 'ping':
                    # Respond to ping
                    pong_msg = {"type": "pong"}
                    await websocket.send(json.dumps(pong_msg))
                    print(f"[{timestamp}] ← PING (sent PONG)")
                
                elif msg_type == 'session_progress':
                    progress = data.get('data', {}).get('progress', 0)
                    status = data.get('data', {}).get('status', 'unknown')
                    print(f"[{timestamp}] ← PROGRESS: {progress:.1f}% ({status})")
                
                elif msg_type == 'trade_executed':
                    trade = data.get('data', {}).get('trade', {})
                    print(f"[{timestamp}] ← TRADE: {trade}")
                
                elif msg_type == 'equity_update':
                    equity = data.get('data', {}).get('equity', {})
                    print(f"[{timestamp}] ← EQUITY: {equity}")
                
                elif msg_type == 'session_completed':
                    final_equity = data.get('data', {}).get('final_equity', 0)
                    total_trades = data.get('data', {}).get('total_trades', 0)
                    print(f"[{timestamp}] ← COMPLETED: equity={final_equity}, trades={total_trades}")
                    print("\n✓ Session completed, closing connection")
                    break
                
                else:
                    print(f"[{timestamp}] ← {msg_type.upper()}: {data.get('data', {})}")
            
            duration = (datetime.now() - start_time).total_seconds()
            print(f"\n{'='*60}")
            print(f"Test Summary:")
            print(f"  Messages received: {message_count}")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Average rate: {message_count/duration if duration > 0 else 0:.1f} msg/s")
            print(f"{'='*60}")
            
    except websockets.exceptions.WebSocketException as e:
        print(f"✗ WebSocket error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n✓ Test interrupted by user")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


async def test_multiple_connections(session_id: str, num_connections: int = 3):
    """Test multiple concurrent WebSocket connections"""
    print(f"\n{'='*60}")
    print(f"Testing Multiple Connections ({num_connections})")
    print(f"{'='*60}")
    
    async def connect_and_listen(conn_id: int):
        try:
            async with websockets.connect(f"{WS_URL}/{session_id}") as websocket:
                print(f"✓ Connection {conn_id} established")
                
                # Listen for a few messages
                for i in range(5):
                    message = await websocket.recv()
                    data = json.loads(message)
                    print(f"  Conn {conn_id}: {data.get('type')}")
                
                print(f"✓ Connection {conn_id} completed")
        except Exception as e:
            print(f"✗ Connection {conn_id} failed: {e}")
    
    # Create multiple connections
    tasks = [connect_and_listen(i) for i in range(num_connections)]
    await asyncio.gather(*tasks)
    
    print(f"\n✓ Multiple connection test completed")


def main():
    """Run WebSocket tests"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_websocket.py <session_id> [--multi]")
        print("\nExample:")
        print("  python test_websocket.py abc123")
        print("  python test_websocket.py abc123 --multi")
        return 1
    
    session_id = sys.argv[1]
    test_multi = "--multi" in sys.argv
    
    print("\n" + "#"*60)
    print("# WebSocket Client Test")
    print("#"*60)
    
    if test_multi:
        asyncio.run(test_multiple_connections(session_id))
    else:
        asyncio.run(test_websocket_connection(session_id))
    
    return 0


if __name__ == '__main__':
    exit(main())
