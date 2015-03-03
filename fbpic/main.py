"""
Fourier-Hankel Particle-In-Cell (FB-PIC) main file

This file steers and controls the simulation.
"""

class Simulation(object) :
    """
    Simulation class that contains all the simulation data,
    as well as the method to perform the PIC cycle.

    Attributes
    ----------
    - fld : a Fields object
    - ptcl : a list of Particles objects

    Methods
    -------
    - step : perform n PIC cycles
    """

    def __init__(self, Nz, zmax, Nr, rmax, Nm, dt) :
        """
        Initializes the simulation structures

        Parameters
        ----------
        Nz : int
            The number of gridpoints in z

        zmax : float
            The size of the simulation box along z
            
        Nr : int
            The number of gridpoints in r

        rmax : float
            The size of the simulation box along r

        Nm : int
            The number of azimuthal modes

        dt : float
            The timestep of the simulation
        """
    
        # Initialize the field structure
        self.fld = Fields(...)
        # Fill the values of the interpolation grid
        # ...
        # Convert to spectral space
        self.fld.interp2spec('E')
        self.fld.interp2spec('B')
        
        # Initialize the particle structure
        self.ptcl = [
            Particles(...),  # Electrons
            Particles(...)   # Ions
            ]
        

    def step(self, N=1) :
        """
        Takes N timesteps in the PIC cycle
    
        The structures fld (Fields object) and ptcl (Particles object)
        have to be defined at that point
    
        Parameter
        ---------
        N : int, optional
            The number of timesteps to take
        """

        # Shortcuts
        ptcl = self.ptcl
        fld = self.fld
        
        # Loop over timesteps
        for _ in range(N) :
            
            # Gather the fields at t = n dt
            for species in ptcl :
                species.gather( fld.interp )
    
            # Push the particles' positions and velocities to t = (n+1/2) dt
            for species in ptcl :
                species.push_p()
                species.halfpush_x()
            # Get the current on the interpolation grid,
            # and then on the spectral grid
            for species in ptcl :
                species.deposit( fld.interp, 'J' )
            fld.interp2spect('J')

            # Push the particles' position to t = (n+1) dt
            for species in ptcl :
                species.halfpush_x()
            # Get the charge density on the interpolation grid,
            # and then on the spectral grid
            for species in ptcl :
                species.deposit( fld.interp, 'rho' )
            fld.interp2spect('rho')
    
            # Push the fields in time
            fld.push()
            # Bring them back to the interpolation grid
            fld.spect2interp('E')
            fld.spect2interp('B')
    
            # Boundary conditions could potentially be implemented here, 
            # on the interpolation grid. This would impose
            # to then convert the fields back to the spectral space.