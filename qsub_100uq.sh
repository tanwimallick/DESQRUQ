#!/bin/bash                                                                                                                                     
#COBALT -n 50                                                                                                
#COBALT -t 12:00:00  
#COBALT -A hpcbdsm                                                                                                                          

source ~/.bashrc

source activate /lus/theta-fs0/projects/hpcbdsm/cooley/gcntf2

echo "Running Cobalt Job $COBALT_JOBID."

mpirun -np 100 -ppn 2 python script100uq.py

#wait



