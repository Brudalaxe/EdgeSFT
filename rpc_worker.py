# Worker Script (Workstation - 192.168.42.118)
import torch
import torch.distributed.rpc as rpc
import os
import logging
import sys
import traceback
import socket
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_connection():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        logging.info("Testing connection to master")
        sock.connect(('192.168.42.236', 29500))  # Updated to Pi6's IP
        logging.info("Successfully connected to master")
        sock.close()
        return True
    except Exception as e:
        logging.error(f"Connection test failed: {e}")
        return False

def main():
    try:
        logging.info("Starting worker node initialization")
        logging.info(f"Python version: {sys.version}")
        logging.info(f"PyTorch version: {torch.__version__}")
        
        # Clear and set environment variables
        os.environ.clear()
        os.environ['MASTER_ADDR'] = '192.168.42.236'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['TP_SOCKET_IFNAME'] = 'eno1'
        os.environ['GLOO_SOCKET_IFNAME'] = 'eno1'
        
        logging.info(f"Environment variables set:")
        logging.info(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
        logging.info(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
        
        logging.info("Waiting 10 seconds for master to start...")
        time.sleep(10)  # Give time to start the master script

        # Test connection to master
        if not test_connection():
            raise Exception("Connection test to master failed")

        logging.info("Configuring RPC options")
        options = rpc.TensorPipeRpcBackendOptions(
            init_method=f'tcp://192.168.42.236:29500',
            num_worker_threads=16,
            _transports=['uv'],
            rpc_timeout=120
        )
        
        logging.info("Attempting RPC initialization")
        rpc.init_rpc(
            "worker",
            rank=1,
            world_size=2,
            rpc_backend_options=options
        )
        
        logging.info("Worker RPC initialized successfully")
        logging.info("Waiting for commands from master...")
        
    except Exception as e:
        logging.error("Exception occurred:")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"Error message: {str(e)}")
        logging.error("Traceback:")
        logging.error(traceback.format_exc())
    finally:
        if torch.distributed.rpc.api._is_current_rpc_agent_set():
            logging.info("Shutting down RPC")
            rpc.shutdown()
            logging.info("RPC shutdown complete")

if __name__ == "__main__":
    main()