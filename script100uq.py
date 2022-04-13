from mpi4py import MPI
import os
import socket
import subprocess


hostname = socket.gethostname()
rank = MPI.COMM_WORLD.Get_rank()
gpu_device = rank % 2
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)

arg = '--config_filename=data100/data0' + str(rank) +'/model/dcrnn_la.yaml &> output' +str(rank)+ '.out'

cmd = "python dcrnn_train_pytorch.py " + arg 
subprocess.run(cmd, shell=True)





