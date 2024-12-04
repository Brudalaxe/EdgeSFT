import os
import torch.distributed.rpc as rpc

def echo_message(message):
    print(f"Received message: {message}")
    return f"Server received: {message}"

def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = '192.168.42.118'  # wkstn IP address
    os.environ['MASTER_PORT'] = '29501'  # Port for communication

    print("Initializing RPC on server...")
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
    
    if rank == 0:
        print("Server is running...")

    rpc.shutdown()
    print("RPC shutdown on server.")

if __name__ == "__main__":
    run_worker(rank=0, world_size=2)