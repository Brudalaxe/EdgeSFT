import torch
import torch.distributed.rpc as rpc
import os
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    print("\nCUDA Debug (cloud.py start):")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")

    os.environ.clear()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error messages
    os.environ['MASTER_ADDR'] = '192.168.42.149'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['TP_SOCKET_IFNAME'] = 'eno1'
    os.environ['GLOO_SOCKET_IFNAME'] = 'eno1'
    
    print("\nCUDA Debug (after env vars):")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")

    logging.info("Initializing cloud node...")
    
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=300,
        init_method=f'tcp://192.168.42.149:29500',
        _transports=['uv']
    )

    try:
        logging.info("Starting RPC initialization...")
        rpc.init_rpc(
            "cloud",
            rank=1,
            world_size=2,
            rpc_backend_options=options
        )
        logging.info("Cloud RPC initialized successfully")
        
        while True:
            pass
            
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Shutting down cloud node...")
    finally:
        if torch.distributed.rpc.api._is_current_rpc_agent_set():
            rpc.shutdown()