# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines a set of functions that are useful when converting the
fields from interpolation grid to the spectral grid and vice-versa
"""
from numba import cuda
import cupy

# ------------------
# Copying functions
# ------------------

@cupy.fuse()
def cuda_copy_2dC_to_2dR_fuse( array_in, array_out_r, array_out_i ) :
    
    array_out_r[:] = cupy.real(array_in)
    array_out_i[:] = cupy.imag(array_in)

def cuda_copy_2dC_to_2dR( array_in, array_out ) :
    """
    Store the complex Nz x Nr array `array_in`
    into the real 2Nz x Nr array `array_out`,
    by storing the real part in the first Nz elements of `array_out` along z,
    and the imaginary part in the next Nz elements.

    Parameters :
    ------------
    array_in: 2darray of complexs
        Array of shape (Nz, Nr)
    array_out: 2darray of reals
        Array of shape (2*Nz, Nr)
    """
    # Set up cuda grid
    Nz, Nr = array_in.shape

    # Copy from array_in to array_out
    # array_out[:Nz,:] = array_in.real
    # array_out[Nz:,:] = array_in.imag
    
    cuda_copy_2dC_to_2dR_fuse(array_in, array_out[:Nz,:], array_out[Nz:,:])

@cupy.fuse()
def cuda_copy_2dR_to_2dC_fuse ( array_in_r, array_in_i, array_out ) :
    
    array_out[:] = array_in_r + 1.j * array_in_i
    
def cuda_copy_2dR_to_2dC( array_in, array_out ) :
    """
    Reconstruct the complex Nz x Nr array `array_out`,
    from the real 2Nz x Nr array `array_in`,
    by interpreting the first Nz elements of `array_in` along z as
    the real part, and the next Nz elements as the imaginary part.

    Parameters :
    ------------
    array_in: 2darray of reals
        Array of shape (2*Nz, Nr)
    array_out: 2darray of complexs
        Array of shape (Nz, Nr)
    """
    # Set up cuda grid
    Nz, Nr = array_out.shape

    # Copy from array_in to array_out
    #array_out[:,:] = array_in[:Nz,:] + 1.j*array_in[Nz:,:]
    
    cuda_copy_2dR_to_2dC_fuse(array_in[:Nz,:], array_in[Nz:,:], array_out)

# ----------------------------------------------------
# Functions that combine components in spectral space
# ----------------------------------------------------

@cupy.fuse()
def cuda_rt_to_pm( buffer_r, buffer_t, buffer_p, buffer_m ) :
    """
    Combine the arrays buffer_r and buffer_t to produce the
    arrays buffer_p and buffer_m, according to the rules of
    the Fourier-Hankel decomposition (see associated paper)
    """

    # Use intermediate variables, as the arrays
    # buffer_r and buffer_t may actually point to the same
    # object as buffer_p and buffer_m, for economy of memory
    value_r = cupy.copy(buffer_r)
    value_t = cupy.copy(buffer_t)
    # Combine the values
    buffer_p[:] = 0.5*( value_r - 1.j*value_t )
    buffer_m[:] = 0.5*( value_r + 1.j*value_t )


@cupy.fuse()
def cuda_pm_to_rt( buffer_p, buffer_m, buffer_r, buffer_t ) :
    """
    Combine the arrays buffer_p and buffer_m to produce the
    arrays buffer_r and buffer_t, according to the rules of
    the Fourier-Hankel decomposition (see associated paper)
    """

    # Use intermediate variables, as the arrays
    # buffer_r and buffer_t may actually point to the same
    # object as buffer_p and buffer_m, for economy of memory
    value_p = cupy.copy(buffer_p)
    value_m = cupy.copy(buffer_m)
    # Combine the values
    buffer_r[:] =  ( value_p + value_m )
    buffer_t[:] =  1.j*( value_p - value_m )
