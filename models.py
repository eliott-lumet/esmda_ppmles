# -*- coding: utf-8 -*-
"""
This file contains the definition of the POD--GPR surrogate model [1] of an LES
model of passive scalar transport in an urban canopy which reproduce the
MUST field experiment.
This surrogate model is trained on the PPMLES [2] dataset of pre-computed LES 
to predict the time-averaged concentration field for new inlet flow direction
and friction velocity. 

[1] Lumet, E., Rochoux, M. C., Jaravel, T., and Lacroix, S. (2025).
Uncertainty-Aware Surrogate Modeling for Urban Air Pollutant Dispersion
Prediction. Building and Environment, page 112287. 
[2] Lumet, E., Jaravel, T., and Rochoux, M. C. (2024). 
PPMLES – Perturbed-Parameter ensemble of MUST Large-Eddy Simulations. Dataset.

@author: lumet
"""
# =============================================================================
# Imports
# ============================================================================= 
import h5py
import joblib as jb
import numpy as np
import pkg_resources

from auxiliaries import min_max_normalize

# =============================================================================
# Class
# ============================================================================= 
class MustSurrogate():
    """
    MUST Surrogate model class.

    Model parameters
    ----------
    ustar: float
        Friction velocity of the flow (m.s^-1). Defines the inlet velocity 
        vertical profile.
    alpha_inlet: float
        Angle between the mean flow direction and the x-axis (°). 
        #TODO: remove
    database: str, {'full', 'train', 'first_ensemble_full', 'first_ensemble_train', 'perso'} OR iterable, optional
       
        The default is 'full'.
    """
    def __init__(self, alpha_inlet=None, ustar=None):
        # General attributes
        self.avg_state = None
        self.pod_coeffs = None

        # Input space definition 
        self._alpha_min = -90.
        self._alpha_max = 30.
        self._ustar_min = 0.0740
        self._ustar_max = 0.8875

        # Inputs reading
        if ustar is None:
            self.ustar = (self._ustar_max - self._ustar_min)/2
        else:
            self.ustar = ustar
        
        if alpha_inlet is None:
            self.alpha_inlet = (self._alpha_max - self._alpha_min)/2
        else:
            self.alpha_inlet = alpha_inlet

        self.parameters = (self.alpha_inlet, self.ustar)

        #  Normalize inputs
        self.normalized_parameters = np.array([min_max_normalize(self.alpha_inlet, self._alpha_min, self._alpha_max), 
                                               min_max_normalize(self.ustar, self._ustar_min, self._ustar_max)])

        # Load offline data
        self.load_mesh()
            
        # Load pre-trained PODs and GPRs for 'POD_GPRs' or solution database for the other on-the-fly surrogate
        self.load_POD_GPR_models()

    # Setters:        
    def set_parameters(self, alpha_inlet, ustar):
        self.ustar = ustar
        self.alpha_inlet = alpha_inlet
        self.parameters = (alpha_inlet, ustar)
        self.normalized_parameters = np.array((min_max_normalize(self.alpha_inlet, self._alpha_min, self._alpha_max), 
                                               min_max_normalize(self.ustar, self._ustar_min, self._ustar_max)))

    # Getters
    def get_mesh_nodes(self):
        """
        Returns the mesh nodes coordinates.
        """
        return np.array([self.mesh['Coordinates']['x'], 
                         self.mesh['Coordinates']['y'],
                         self.mesh['Coordinates']['x']]).T
    
    def get_mesh_conns(self):
        """
        Returns the mesh connectivities.
        """
        return self.mesh['Connectivity']['tet->node']

    def get_mesh_volumes(self):
        """
        Returns the mesh cells volumes.
        """
        return self.mesh['VertexData']['volume']

    def get_pod_coeffs(self):
        """
        Returns the POD coefficients estimated by the GPs
        """
        return self.pod_coeffs

    def get_pod_coeffs_std(self):
        """
        Returns the variance of the POD coefficients estimated by the GPs.
        """
        return self.pod_coeffs_std

    def get_std_state(self):
        """
        Returns the standard deviation of the estimated field distribution.
        """
        return self.std_state

    # Loading functions    
    def load_mesh(self): 
        """
        Load the solution mesh on which the fields of the database
        are defined.
        """         
        self.mesh = h5py.File(pkg_resources.resource_filename(__name__, 'data/models/Turbo_MUST_analysis.mesh.h5'), 'r')
        self.n_nodes = np.shape(self.mesh['Coordinates']['x'])[0]

    def load_POD_GPR_models(self): 
        """
        Load the two PAD models and associated GPRs pre-trained models with the
        normalized database.
        """        
        # Load the scaler and POD models and pretrained GPRs
        self._output_scaler_log = jb.load(pkg_resources.resource_filename(__name__, 'data/models/fields_log_scaler_1D.joblib'))
        self._pca_log = jb.load(pkg_resources.resource_filename(__name__, 'data/models/pca.joblib'))
        self._noisyGPRs_log = jb.load(pkg_resources.resource_filename(__name__, 'data/models/gaussian_processes.joblib')).set_params(n_jobs=1)

    # Checking function
    def check_parameters(self):
        """
        Check if the model parameters are withing their range of definition.
        """
        if self.alpha_inlet < self._alpha_min:
            raise Exception(f'ERROR: the parameter alpha_inlet is smaller than its minimal value: {self.alpha_inlet:.1f}° < {self._alpha_min:.1f}°')
        elif self.alpha_inlet > self._alpha_max:
            raise Exception(f'ERROR: the parameter alpha_inlet is larger than its maximal value: {self.alpha_inlet:.1f}° > {self._alpha_max:.1f}°')

    # Prediction functions
    def GP_log_predict(self):
        """
        Predict the n_modes pod coefficients with the n_modes independent
        Gaussian Process Regressors based on the POD-log decomposition.
        """ 
        return self._noisyGPRs_log.predict(self.normalized_parameters[0].reshape(1,-1), return_std=True)[0]

    def GP_log_sample(self, n_samples, random_state=None):
        """
        Predict n_samples realizations of the the n_modes pod coefficients with
        the n_modes independent Gaussian Process Regressors based on the 
        POD-log decomposition.
        """
        return self._noisyGPRs_log.sample_y(self.normalized_parameters[0].reshape(1,-1), n_samples=n_samples, random_state=random_state)[:,0]

    # Model computation:
    def forecast(self, alpha_inlet=None, ustar=None):  
        # Parameters reading
        if ustar != None:
            self.ustar = ustar

        if alpha_inlet != None:
            self.alpha_inlet = alpha_inlet
        
        self.parameters = (self.alpha_inlet, self.ustar)
        self.check_parameters()
        self.normalized_parameters = np.array((min_max_normalize(self.alpha_inlet, self._alpha_min, self._alpha_max), 
                                               min_max_normalize(self.ustar, self._ustar_min, self._ustar_max)))

        # GPs prediction
        gpr_log_estimates = self.GP_log_predict()
        self.pod_coeffs = gpr_log_estimates[0]
        self.pod_coeffs_std = gpr_log_estimates[1]
        
        # POD inverse transform
        self.avg_state = self._output_scaler_log.inverse_transform(self._pca_log.inverse_transform(self.pod_coeffs))

        # Filter negative values
        self.avg_state[self.avg_state<0] = 0
    
    # Model sampling
    def sample_y(self, n_samples, random_state=None, alpha_inlet=None, ustar=None):  
        """
        Generate one prediction realization given the posterior distribution
        of the POD--GPRs
        """
        # Parameters reading
        if ustar != None:
            self.ustar = ustar

        if alpha_inlet != None:
            self.alpha_inlet = alpha_inlet
        
        self.parameters = (self.alpha_inlet, self.ustar)
        self.check_parameters()
        self.normalized_parameters = np.array((min_max_normalize(self.alpha_inlet, self._alpha_min, self._alpha_max), 
                                               min_max_normalize(self.ustar, self._ustar_min, self._ustar_max)))


        # GPRs realizations
        gp_log_samples = self.GP_log_sample(n_samples, random_state)
            
        # POD inverse transform
        c_samples = self._output_scaler_log.inverse_transform(self._pca_log.inverse_transform(gp_log_samples))/self.ustar

        # Filter negative values
        c_samples[c_samples<0] = 0

        return c_samples