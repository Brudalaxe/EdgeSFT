# Master Script (Pi2 - 192.168.42.151)
import torch
import torch.distributed.rpc as rpc
import os
import logging
import socket
import time

logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def test_binding():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        # First try to connect to check if port is in use
        result = sock.connect_ex(('127.0.0.1', 29500))
        if result == 0:
            logging.error("Port 29500 is already in use")
            return False
        
        # Try binding
        sock.bind(('0.0.0.0', 29500))
        sock.listen(1)
        logging.info("Successfully bound to port 29500")
        sock.close()
        
        # Small delay to ensure socket is properly closed
        time.sleep(1)
        return True
    except Exception as e:
        logging.error(f"Port binding test failed: {e}")
        return False
    finally:
        sock.close()

def main():
    logging.info("Starting master initialization")
    
    # Kill any existing processes using port 29500
    os.system("sudo fuser -k 29500/tcp")
    time.sleep(1)
    
    os.environ.clear()
    os.environ['MASTER_ADDR'] = '192.168.42.151'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['TP_SOCKET_IFNAME'] = 'wlan0'
    os.environ['GLOO_SOCKET_IFNAME'] = 'wlan0'

    logging.info(f"Environment variables:")
    for key in ['MASTER_ADDR', 'MASTER_PORT', 'TP_SOCKET_IFNAME', 'GLOO_SOCKET_IFNAME']:
        logging.info(f"{key}: {os.environ.get(key)}")

    if not test_binding():
        raise Exception("Failed to bind to port 29500")

    logging.info("Configuring RPC options")
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=120,
        init_method=f'tcp://192.168.42.151:29500',
        _transports=['uv']
    )

    try:
        logging.info("Attempting RPC initialization")
        rpc.init_rpc(
            "master",
            rank=0,
            world_size=2,
            rpc_backend_options=options
        )
        logging.info("RPC initialization successful")
        
        logging.info("Testing computation")
        result = rpc.rpc_sync(
            "worker",
            torch.add,
            args=(torch.ones(2), 1)
        )
        logging.info(f"Computation result: {result}")
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise
    finally:
        if torch.distributed.rpc.api._is_current_rpc_agent_set():
            rpc.shutdown()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Shutting down master...")
    finally:
        # Ensure port is freed
        os.system("sudo fuser -k 29500/tcp")