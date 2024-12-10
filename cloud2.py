import torch
import torch.distributed.rpc as rpc
import os
import logging
from model_classes import CloudWorker, MidiBertBack

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    os.environ.clear()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['MASTER_ADDR'] = '192.168.42.236'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['TP_SOCKET_IFNAME'] = 'eno1'
    os.environ['GLOO_SOCKET_IFNAME'] = 'eno1'
    
    device = torch.device("cuda:0")

    logging.info("Initialising cloud node...")
    
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=600,
        init_method=f'tcp://192.168.42.236:29500',
        _transports=['uv']
    )
    
    options.set_device_map("edge", {"cpu": "cpu"})

    try:
        logging.info("Starting RPC initialisation...")
        rpc.init_rpc(
            "cloud",
            rank=1,
            world_size=2,
            rpc_backend_options=options
        )
        logging.info("Cloud RPC initialised successfully")
        
        worker = CloudWorker()
        
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