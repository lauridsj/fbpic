# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of generic functions that operate on a GPU.
"""
from numba import cuda

# Check if CUDA is available and set variable accordingly
try:
    cuda_installed = cuda.is_available()
except Exception:
    cuda_installed = False

if cuda_installed:
    # Infer if GPU is P100 or V100 or other
    if "P100" in str(cuda.gpus[0]._device.name):
        cuda_gpu_model = "P100"
    elif "V100" in str(cuda.gpus[0]._device.name):
        cuda_gpu_model = "V100"
    else:
        cuda_gpu_model = "other"

try:
    import cupy
    cupy_installed = cupy.is_available()
    cupy_major_version = int(cupy.__version__[0])
except (ImportError, AssertionError):
    cupy_installed = False
    cupy_major_version = None

# -----------------------------------------------------
# CUDA grid utilities
# -----------------------------------------------------

def cuda_tpb_bpg_1d(x, TPB = 256):
    """
    Get the needed blocks per grid for a 1D CUDA grid.

    Parameters :
    ------------
    x : int
        Total number of threads

    TPB : int
        Threads per block

    Returns :
    ---------
    BPG : int
        Number of blocks per grid

    TPB : int
        Threads per block.
    """
    # Calculates the needed blocks per grid
    BPG = int(x/TPB + 1)
    return BPG, TPB

def cuda_tpb_bpg_2d(x, y, TPBx = 1, TPBy = 128):
    """
    Get the needed blocks per grid for a 2D CUDA grid.

    Parameters :
    ------------
    x, y  : int
        Total number of threads in first and second dimension

    TPBx, TPBy : int
        Threads per block in x and y

    Returns :
    ------------
    (BPGx, BPGy) : tuple of ints
        Number of blocks per grid in x and y

    (TPBx, TPBy) : tuple of ints
        Threads per block in x and y.
    """
    # Calculates the needed blocks per grid
    BPGx = int(x/TPBx + 1)
    BPGy = int(y/TPBy + 1)
    return (BPGx, BPGy), (TPBx, TPBy)

# -----------------------------------------------------
# CUDA memory management
# -----------------------------------------------------

def send_data_to_gpu(simulation):
    """
    Send the simulation data to the GPU.
    Calls the functions of the particle and field package
    that send the data to the GPU.

    Parameters :
    ------------
    simulation : object
        A simulation object that contains the particle
        (ptcl) and field object (fld)
    """
    # Send particles to the GPU (if CUDA is used)
    for species in simulation.ptcl :
        if species.use_cuda:
            species.send_particles_to_gpu()
    # Send fields to the GPU (if CUDA is used)
    simulation.fld.send_fields_to_gpu()

def receive_data_from_gpu(simulation):
    """
    Receive the simulation data from the GPU.
    Calls the functions of the particle and field package
    that receive the data from the GPU.

    Parameters :
    ------------
    simulation : object
        A simulation object that contains the particle
        (ptcl) and field object (fld)
    """
    # Receive the particles from the GPU (if CUDA is used)
    for species in simulation.ptcl :
        if species.use_cuda:
            species.receive_particles_from_gpu()
    # Receive fields from the GPU (if CUDA is used)
    simulation.fld.receive_fields_from_gpu()

class MoveSimToGpuIfNeeded(object):
    """
    Context manager that temporarily moves the simulation data to the GPU,
    if the data is originally on the CPU when entering the context manager
    """

    def __init__(self, simulation):
        """
        Initialize the context manager

        Parameters:
        -----------
        simulation: object
            A simulation object that contains the particle
            (ptcl) and field object (fld)
        """
        # Check whether the data is initially on the CPU or GPU
        self.fields_were_on_gpu = simulation.fld.data_is_on_gpu
        self.species_were_on_gpu = [ species.data_is_on_gpu \
                                     for species in simulation.ptcl ]
        # Keep a reference to the simulation
        self.sim = simulation

    def __enter__(self):
        """
        Move the data to the GPU (if it was originally on the CPU)
        """
        if self.sim.use_cuda:
            if not self.fields_were_on_gpu:
                self.sim.fld.send_fields_to_gpu()
            for i, species in enumerate(self.sim.ptcl):
                if not self.species_were_on_gpu[i]:
                    species.send_particles_to_gpu()

    def __exit__(self, type, value, traceback):
        """
        Move the data back to the CPU (if it was originally on the CPU)
        """
        if self.sim.use_cuda:
            if not self.fields_were_on_gpu:
                self.sim.fld.receive_fields_from_gpu()
            for i, species in enumerate(self.sim.ptcl):
                if not self.species_were_on_gpu[i]:
                    species.receive_particles_from_gpu()


# -----------------------------------------------------
# CUDA mpi management
# -----------------------------------------------------

def mpi_select_gpus(mpi):
    """
    Selects the correct GPU used by the current MPI process

    Parameters :
    ------------
    mpi: an mpi4py.MPI object
    """
    n_gpus = len(cuda.gpus)
    rank = mpi.COMM_WORLD.rank
    for i_gpu in range(n_gpus):
        if rank%n_gpus == i_gpu:
            cuda.select_device(i_gpu)
        mpi.COMM_WORLD.barrier()
        

# -----------------------------------------------------
# CUDA kernel decorator
# -----------------------------------------------------

if cupy_installed:
    
    def compile_cupy_static(argtypes=[]):
    
        class CupyKernel(object):

            def __init__(self, func):

                numba_kernel = cuda.jit(argtypes=argtypes)(func)

                module = cupy.cuda.function.Module()
                module.load( bytes( numba_kernel.ptx, 'UTF-8' ) )
                self.cupy_kernel = module.get_function(numba_kernel.entry_name)           

            def __call__(self, bpg, tpb, *args):

                kernel_args = []
                for a in args:
                    if isinstance(a, cupy.ndarray):
                        kernel_args.extend([0, 0, a.size, a.dtype.itemsize, a, *a.shape, *a.strides])
                    else:
                        kernel_args.append(a)

                self.cupy_kernel(bpg, tpb, kernel_args)

        return CupyKernel
    
    
    
    class compile_cupy(object):
        
        def __init__(self, func):
            
            self.compiled = False
            self.python_func = func
          
        def __getitem__(self, bt):
            
            def call_kernel(*args):
                
                if not self.compiled:

                    numba_kernel = cuda.jit()(self.python_func).specialize(*args)

                    module = cupy.cuda.function.Module()
                    module.load( bytes( numba_kernel.ptx, 'UTF-8' ) )
                    self.cupy_kernel = module.get_function(numba_kernel.entry_name)
                    self.compiled = True

                kernel_args = []
                for a in args:
                    if isinstance(a, cupy.ndarray):
                        kernel_args.extend([0, 0, a.size, a.dtype.itemsize, a, *a.shape, *a.strides])
                    else:
                        kernel_args.append(a)

                self.cupy_kernel(bt[0], bt[1], kernel_args)
                
            return call_kernel
                