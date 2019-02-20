from __future__ import print_function
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD

print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))

comm.Barrier()