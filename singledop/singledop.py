"""
Title/Version
-------------
Single Doppler Retrieval Toolkit (SingleDop)
singledop v0.9
Developed & tested with Python 2.7 & 3.4
Last changed 02/09/2016


Author
------
Timothy Lang
NASA MSFC
timothy.j.lang@nasa.gov
(256) 961-7861


Overview
--------
To access this module, install it from the source using the setup.py script.
Then:
import singledop


Notes
-----
Dependencies: numpy, matplotlib, basemap, scipy, math, time, pyart, pytda,
              pickle, warnings, xray/xarray


References
----------
Xu et al., 2006: Background error covariance functions for vector wind analyses
using Doppler-radar radial-velocity observations. Q. J. R. Meteorol. Soc., 132,
2887-2904.


Change Log
----------
v0.9 Changes (02/09/16):
1. Added ability to filter retrievals far from observations, via filter_data &
   filter_distance keywords. These values are masked.
2. Updated import statements to use xarray if available.
3. Added additional title text to AnalysisDisplay.four_panel_plot() method.

v0.8.1 Changes (11/30/15):
1. Added common sub-module with radar_coords_to_cart function that Py-ART
   just removed.

v0.8 Changes (08/31/15):
1. Fixed issues for when radar object fields lack masks or fill values.
2. Added ability to select display sweep in AnalysisDisplay.four_panel_plot()

v0.7 Changes (08/03/15):
1. Made code compatible with Python 3.

v0.6 Changes (07/02/15):
1. Made code pep8 compliant.

v0.5 Changes (05/05/15):
1. Added NetcdfSave class that uses xray to save/load analysis object
   to/from netCDF.
2. Created BaseAnalysis and SimpleObject helper classes to assist
   with refactoring.
3. Refactored AnalysisDisplay.four_panel_plot() to make it more intelligible.
4. Added VAR_LIST global variable to store the names of most commonly shared
   SingleDoppler2D attributes. Loop over this list to simplify attribute
   assignments.
5. Changed reflectivity colormap to one available in pyart.

v0.4 Changes (01/15/15):
1. Added compute_vad_ring() and get_u_and_v_from_ws_and_wd() independent
   functions. New SingleDoppler2D.analyze_vad_rings() method uses these
   functions to compute VAD analyses at several ranges. The medians from this
   analysis are used as the scalar Ub and Vb background winds. To use VAD to
   get background wind estimates, set the use_vad flag in
   SingleDoppler2D.__init__().
2. Renamed some SingleDoppler2D methods to make their purpose clearer.
3. More documentation added to various functions and methods.

v0.3 Changes (12/29/14):
1. Added capability of ingesting user-specified background field via Ub & Vb
   keywords when doing analysis of real radar data. Currently Ub and Vb can
   only be scalars.
2. Moved SingleDoppler2D.compute_beta_and_m() to background wind routines to
   avoid the need for duplicate code.

v0.2 Changes (12/16/2014):
1. Added AnalysisDisplay class to take the plotting load off SingleDoppler2D.
   AnalysisDisplay has the following plotting methods: plot_velocity_vectors(),
   plot_velocity_contours(), plot_radial_tangential_contours(), and
   four_panel_plot(). The latter two have hard-coded figure/axes creation,
   while the former two are more flexible (similar to RadarDisplay plotting
   methods in Py-ART) and are used by the canned routines. All plotting
   methods have more options available now.
3. Added SaveFile class to enable saving/loading of AnalysisDisplay-relevant
   information. This allows the user to avoid rerunning old retrievals.

v0.1 Functionality (12/12/2014):
1. Input simulated wind field or real radar observations of Doppler velocity.
   Retrieve 2D wind field on the conical surface of the radar beam.
2. Basic plotting (velocity contours, velocity vectors) supported.


Planned Updates
---------------
1. Refactoring to reduce function passing of variables
2. Add capability to determine L from data instead of specifying it
3. Add capability to handle more complicated functions for covariance matrix
   elements; i.e., incl. a, b != 0, 1 in Xu et al. (2006) Eq. 2.5.

"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import scipy
import math
import time
import warnings
import pickle
import pyart
from pytda import get_sweep_data, get_sweep_azimuths, get_sweep_elevations, \
                  flatten_and_reduce_data_array
from .common import radar_coords_to_cart
from .cmap_map import lighten_cmap
try:
    import xarray as xray
except ImportError:
    try:
        import xray
    except ImportError:
        warnings.warn(
            'xray/xarray not installed, save using SaveFile (pickle)')

##############################

VERSION = '0.9'

# Hard coding of constants & default parameters
DEFAULT_L = 30.0  # km
DEFAULT_SIGMA = 10.0  # m/s
DEFAULT_SIGMA_OBS = 1.0  # m/s
DEFAULT_GRID_SPACING = 1.0  # km
DEFAULT_GRID_EDGE = 60.0  # km
DEFAULT_VR = 'velocity'
DEFAULT_DZ = 'reflectivity'
re = 6371.1  # km
RNG_MULT = 1000.0  # m per km
DZ_CMAP = lighten_cmap(cm.get_cmap('pyart_LangRainbow12'))
VR_CMAP = 'bwr'
DEFAULT_LABELS = ['Distance E-W (km)', 'Distance N-S (km)']
DEFAULT_LEVELS = -24.0 + 4.0 * np.arange(13)
BAD_DATA_VAL = -32768
VAR_LIST = ['analysis_x', 'analysis_y', 'analysis_vr', 'analysis_vt',
            'analysis_u', 'analysis_v', 'grid_limits', 'L']

##############################


class SingleDoppler2D(object):

    """
    Retrieve low-level 2D winds from either real or simulated radar
    observations. We work in two spaces: observation and analysis.
    Observation: The space where the observations are located
    Analysis: The space defined by the analysis grid
    Both are on the 2-D conical radar sweep surface

    Attributes
    ----------
    grid_spacing, grid_edge = Spacing, edge info to construct analysis grid
    sigma, sigma_obs = Standard deviation of background winds, observed winds
    L = Decorrelation length scale
    analysis_x, analysis_xf = 2D, 1D analysis grid locations along x-axis
    analysis_y, analysis_yf = 2D, 1D analysis grid locations along y-axis
    analysis_Beta = 1D non-radar-convention angle at analysis locations (0 = E)
    obs_xf, obs_yf = 1D arrays of Cartesian locations for observations
    obs_vr, obs_vrf = 2D, 1D versions of observed Doppler radial velocity
    obs_Beta = 1D non-radar-convention angle at observation locations (0 = E)
    radar = Py-ART radar object
    U, V = Simulated U and V winds (need to be on analysis grid)
    analysis_vrb, analysis_vtb = Radial, tangential background winds (analysis)
    obs_vrbf = 1D background winds converted to radial velocity from radar
    obs_Crr = Error covariance matrix in observation space
    M, N = Observation matrix size, analysis grid dimension (both are square)
    analysis_vr, analysis_vt = Retrieved radial, tangential analysis winds
    analysis_u, analysis_v = Retrieved U, V winds on analysis grid
    number_of_beams, ngates = # of beams, gates used for simulated radar obs
    max_range = Max rng to consider in analysis (can be > analysis grid extent)
    az_spacing, range_spacing = Azimuth, range spacing for simulated radar data
    azimuth, slant_range = 2D azimuth, range info for simulated observations
    delta_vr, delta_vt = Retrieved radial, tang. increments to bkgrnd winds
    z_vector = Solution state vector from solved linear system
    analysis_Ub, analysis_Vb = 2D background winds on analysis grid
    obs_Ub, obs_Vb = Scalar or 1D background winds in observation space
    vad_ws, vad_wd = 1D VAD wind speed and direction as function of range
    vad_u, vad_v = Median VAD-derived U, V winds on sweep (scalars)
    range_rings = 1D array of ranges used for VAD analysis
    filter_data = Boolean controlling whether distance retrievals are filtered.
    filter_distance = Distance (km) beyond which retrieved data are filtered.
    """

    def __init__(self, radar=None, sweep_number=0,
                 grid_spacing=DEFAULT_GRID_SPACING,
                 grid_edge=DEFAULT_GRID_EDGE, sigma=DEFAULT_SIGMA,
                 sigma_obs=DEFAULT_SIGMA_OBS, L=DEFAULT_L, U=10.0, V=10.0,
                 Ub=None, Vb=None, name_vr=DEFAULT_VR, noise=True,
                 az_spacing=2.0, use_vad=True, verbose=False,
                 range_spacing=1.0, range_limits=None, azimuth_limits=None,
                 max_range=100.0, xgrid=None, ygrid=None, thin_factor=[2, 4],
                 filter_data=False, filter_distance=None):
        """
        Initializes class based on user-specified information. If user provides
        Py-ART radar object, the analysis will be performed on the real data.

        If user does not provide a radar object, then the analysis will be
        performed using user-specified (or default) info about simulated wind
        field.

        Arguments
        ---------
        radar = Py-ART radar object
        sweep_number = Sweep to consider
        grid_spacing = Resolution of analysis grid (km)
        grid_edge = Edge of analysis grid (grid boundaries = +/- grid_edge)
        sigma = Standard deviation of background wind error (wind speed units)
        sigma_obs = Standard deviation of radial velocity observation error
        L = Decorrelation length scale (km)
        U, V = Synthetic U, V winds (for simulated data only)
        Ub, Vb = User specified background U, V winds (normally scalars)
        name_vr = Py-ART field name used for radial velocity
        noise = Flag to turn on/off Gaussian noise in synthetic radar obs
        az_spacing = Spacing used to develop synthetic radar obs (deg)
        use_vad = Flag to do VAD analysis on real radar obs
        verbose = Flag to provide additional text updates
        range_spacing = Spacing used to develop synthetic radar obs (km)
        range_limits = 2-element array to designate analysis range limits (km)
        azimuth_limits = 2-element array to mask azimuths (deg, simulated data)
        max_range = Maximum range to consider in analysis (km)
        xgrid, ygrid = User-specified 1D input grids for each axis (simulated)
        thin_factor = 2-element array of factors to thin azimuth, range
        filter_data = Set to True to filter retrievals in regions far
                      from observations
        filter_distance = Min distance (km) from nearest obs to filter data.
                          Set to L / 2 by default.
        """
        self.populate_analysis_metadata(grid_spacing=grid_spacing,
                                        grid_edge=grid_edge, sigma=sigma,
                                        sigma_obs=sigma_obs, L=L)
        self.filter_data = filter_data
        if filter_distance is None:
            self.filter_distance = self.L / 2
        else:
            self.filter_distance = filter_distance

        if radar is not None:
            # Computation using real radar data
            self.get_radar_data(radar, sweep_number=sweep_number,
                                name_vr=name_vr, max_range=max_range,
                                range_limits=range_limits,
                                thin_factor=thin_factor)
            # Use VAD to obtain background field
            if use_vad:
                self.analyze_vad_rings(
                    field=name_vr, sweep_number=sweep_number,
                    verbose=verbose)
                self.get_obs_background_field(self.vad_u, self.vad_v)
                self.get_analysis_background_field(self.vad_u, self.vad_v)
            # Otherwise specify background field
            else:
                self.get_obs_background_field(Ub, Vb)
                self.get_analysis_background_field(Ub, Vb)
        else:
            # Computation using synthetic wind data
            # Mainly for testing purposes
            if xgrid is None or ygrid is None:
                self.construct_synthetic_grid(az_spacing=az_spacing,
                                              range_spacing=range_spacing,
                                              max_range=max_range)
            else:
                self.use_input_grid(xgrid=xgrid, ygrid=ygrid)
            self.construct_synthetic_U_and_V(U=U, V=V)
            self.get_simulated_radar_data(noise=noise,
                                          range_limits=range_limits,
                                          azimuth_limits=azimuth_limits)
            self.get_obs_background_field(0.0, 0.0)  # Bkgrnd 0 in sims for now
            self.get_analysis_background_field(0.0, 0.0)
        # Actual retrieval done here
        self.compute_single_doppler_retrieval()

    def populate_analysis_metadata(
            self, grid_spacing=DEFAULT_GRID_SPACING,
            grid_edge=DEFAULT_GRID_EDGE, sigma=DEFAULT_SIGMA,
            sigma_obs=DEFAULT_SIGMA_OBS, L=DEFAULT_L):
        """
        Populate grid metadata using user specified information
        Stored as class attributes to simplify function calls
        grid_spacing = Resolution of analysis grid (km)
        grid_edge = Edge of analysis grid (grid boundaries = +/- grid_edge)
        sigma = Standard deviation of background wind error (wind speed units)
        sigma_obs = Standard deviation of radial velocity observation error
        L = Decorrelation length scale (km)
        """
        self.grid_spacing = grid_spacing
        self.grid_edge = np.abs(grid_edge)
        self.sigma = sigma
        self.sigma_obs = sigma_obs
        self.L = L
        self.construct_analysis_grid()

    def construct_analysis_grid(self):
        """Construct the analysis grid using user-specified information"""
        self.N = 1 + np.int32(np.round(2.0*self.grid_edge/self.grid_spacing))
        self.grid_limits = [-1.0*self.grid_edge, 1.0*self.grid_edge]
        x = self.grid_spacing * np.arange(self.N) - self.grid_edge
        y = 1.0 * x
        self.analysis_x, self.analysis_y = np.meshgrid(x, y)
        self.analysis_xf = self.analysis_x.ravel()
        self.analysis_yf = self.analysis_y.ravel()
        self.analysis_Beta = atan2_array(self.analysis_yf, self.analysis_xf)

    def get_radar_data(self, radar, sweep_number=0, name_vr=DEFAULT_VR,
                       max_range=100.0, range_limits=None, thin_factor=[2, 4]):
        """
        If real radar data are used, this method retrieves the flattened vector
        of xy locations of observations as well as the observed Vr values
        radar = Py-ART radar object
        sweep_number = Sweep number to consider
        name_vr = Name of velocity field
        max_range = Maximum range of observations to consider
        range_limits = Interval of ranges to consider
        thin_factor = Multiples used to reduce azimuth [0] and range [1] data
        """
        vr_sweep = get_sweep_data(radar, name_vr, sweep_number)
        try:
            fill_val = radar.fields[name_vr]['_FillValue']
        except KeyError:
            fill_val = BAD_DATA_VAL
        xx, yy = get_x_and_y_from_radar(radar, sweep_number)
        groundr = (xx**2 + yy**2)**0.5
        xx = xx[0::thin_factor[0], 0::thin_factor[1]]
        yy = yy[0::thin_factor[0], 0::thin_factor[1]]
        if range_limits is not None:
            condition_range = np.logical_and(groundr >= np.min(range_limits),
                                             groundr <= np.max(range_limits))
            self.max_range = np.max(range_limits)
        else:
            condition_range = groundr < max_range
            self.max_range = max_range
        condition_vr = vr_sweep != fill_val
        condition = np.logical_and(condition_range, condition_vr)
        condition = condition[0::thin_factor[0], 0::thin_factor[1]]
        condition = condition.ravel()
        vr_sweep = vr_sweep[0::thin_factor[0], 0::thin_factor[1]]
        self.obs_xf = flatten_and_reduce_data_array(xx, condition)
        self.obs_yf = flatten_and_reduce_data_array(yy, condition)
        self.obs_vrf = flatten_and_reduce_data_array(vr_sweep, condition)
        self.radar = radar

    def filter_distant_gridpoints(self):
        """
        Single-Doppler analyses tend to blow up far from real data.
        This method sets retrieved Vr & Vt values to 0 when the distance
        to the closest real data point is > L/2.
        """
        if self.filter_data:
            axf = self.analysis_x.ravel()
            ayf = self.analysis_y.ravel()
            avrf = self.analysis_vr.ravel()
            avtf = self.analysis_vt.ravel()
            for i in np.arange(len(axf)):
                dist = np.sqrt(
                    (self.obs_xf-axf[i])**2 + (self.obs_yf-ayf[i])**2)
                if np.min(dist) > self.filter_distance:
                    avrf[i] = BAD_DATA_VAL
                    avtf[i] = BAD_DATA_VAL
            self.analysis_vr = np.reshape(avrf, (self.N, self.N))
            self.analysis_vt = np.reshape(avtf, (self.N, self.N))
        self.analysis_vr = np.ma.asanyarray(self.analysis_vr)
        self.analysis_vr.mask = self.analysis_vr == BAD_DATA_VAL
        self.analysis_vt = np.ma.asanyarray(self.analysis_vt)
        self.analysis_vt.mask = self.analysis_vt == BAD_DATA_VAL

    def get_simulated_radar_data(self, noise=True, range_limits=None,
                                 azimuth_limits=None):
        """
        Given simulated U and V, now convert to simulated Vr
        Can add Gaussian noise to simulated observations, as well as
        constrain by range and azimuth
        noise = Flag to turn on/off Gaussian noise in synthetic radar obs
        range_limits = 2-element array to designate analysis range limits (km)
        azimuth_limits = 2-element array to mask azimuths (deg, simulated data)
        """
        # Following works for radar convention azimuth (0 deg = North)
        Vr = self.U * np.sin(np.deg2rad(self.azimuth)) + \
            self.V * np.cos(np.deg2rad(self.azimuth))
        if noise:
            errors = np.reshape(np.random.normal(scale=self.sigma_obs,
                                                 size=np.size(self.U)),
                                np.shape(self.U))
        else:
            errors = 0.0
        Vrm = Vr + errors
        self.obs_vr = Vrm
        self.obs_vrf = Vrm.ravel()
        self.obs_xf = self.obs_x.ravel()
        self.obs_yf = self.obs_y.ravel()
        mask = self.create_mask(azimuth_limits=azimuth_limits,
                                range_limits=range_limits)
        if mask is not None:
            self.obs_vrf = self.obs_vrf[mask]
            self.obs_xf = self.obs_xf[mask]
            self.obs_yf = self.obs_yf[mask]
        self.radar = None

    def compute_beta_and_m(self):
        """
        Get observed Beta angles, as well as initialize the observation-space
        C matrix (MxM, where M is number of observations)
        """
        # Beta is in non-radar convention (0 = E)
        self.obs_Beta = atan2_array(self.obs_yf, self.obs_xf)
        self.M = len(self.obs_vrf)
        print(self.M, 'total observations (M)')
        self.obs_Crr = np.zeros((self.M, self.M), 'float')

    def create_mask(self, azimuth_limits=None, range_limits=None):
        """
        Create simulated data mask using input azimuth and range limits
        range_limits = 2-element array to designate analysis range limits (km)
        azimuth_limits = 2-element array to mask azimuths (deg, simulated data)
        """
        if azimuth_limits is not None:
            azimuth_cond = np.logical_and(
                self.azimuth >= np.min(azimuth_limits),
                self.azimuth <= np.max(azimuth_limits))
        else:
            azimuth_cond = None
        if range_limits is not None:
            range_cond = np.logical_and(
                self.slant_range >= np.min(range_limits),
                self.slant_range <= np.max(range_limits))
        else:
            range_cond = None
        if azimuth_cond is None and range_cond is None:
            return None
        if azimuth_cond is not None and range_cond is not None:
            azimuth_cond = azimuth_cond.ravel()
            range_cond = range_cond.ravel()
            return np.logical_and(azimuth_cond, range_cond)
        if azimuth_cond is None and range_cond is not None:
            return range_cond.ravel()
        if azimuth_cond is not None and range_cond is None:
            return azimuth_cond.ravel()

    def compute_single_doppler_retrieval(self):
        """
        Performs single-Doppler retrieval whether input data are real
        or simulated. Matrix equations solved via stock SciPy/NumPy routines
        """
        for index in np.arange(self.M):
            Beta1 = math.atan2(self.obs_yf[index], self.obs_xf[index])
            Ctmp = get_Crr(self.obs_xf, self.obs_yf, self.obs_xf[index],
                           self.obs_yf[index], self.obs_Beta, Beta1,
                           self.L, self.sigma)
            self.obs_Crr[index, :] = Ctmp[:]
        A = self.obs_Crr + self.sigma_obs**2 * np.eye(self.M)
        b = self.obs_vrf - self.obs_vrbf  # Observation innovation vector
        self.z_vector = scipy.linalg.solve(A, b, sym_pos=True)
        # self.z_vector = np.linalg.solve(A, b)  # scipy appears to be faster
        delta_vr = 0.0 * self.analysis_xf
        delta_vt = 0.0 * self.analysis_xf
        for index in np.arange(len(self.analysis_xf)):
            Beta1 = math.atan2(self.analysis_yf[index],
                               self.analysis_xf[index])
            Ctmp = get_Crr(self.analysis_xf[index], self.analysis_yf[index],
                           self.obs_xf, self.obs_yf, self.obs_Beta,
                           self.analysis_Beta[index], self.L, self.sigma)
            delta_vr[index] = np.sum(Ctmp * self.z_vector)
            Ctmp = get_Crt(self.analysis_xf[index], self.analysis_yf[index],
                           self.obs_xf, self.obs_yf, self.obs_Beta,
                           self.analysis_Beta[index], self.L, self.sigma)
            delta_vt[index] = np.sum(Ctmp * self.z_vector)
        self.delta_vr = np.reshape(delta_vr, (self.N, self.N))
        self.delta_vt = np.reshape(delta_vt, (self.N, self.N))
        self.analysis_vr = self.delta_vr + self.analysis_vrb
        # Sign issue correction follows, CCW should be positive
        # Arises from atan2 output not in radar-convention format (0 = N)
        # In radar convention, angle decreases with CCW rotation
        # But for atan2 output angle increases with CCW rotation
        self.delta_vt *= -1.0
        self.analysis_vt = self.delta_vt + self.analysis_vtb
        self.filter_distant_gridpoints()

    def get_velocity_vectors(self):
        """
        Once Vr and Vt are obtained, compute analyzed U & V given xy locations
        """
        Beta = atan2_array(self.analysis_y, self.analysis_x)
        self.analysis_u = self.analysis_vr * np.cos(Beta) + \
            self.analysis_vt * np.cos(Beta+np.pi/2.0)
        self.analysis_v = self.analysis_vr * np.sin(Beta) + \
            self.analysis_vt * np.sin(Beta+np.pi/2.0)

    def construct_synthetic_U_and_V(self, U=10.0, V=10.0):
        """
        Construct U and V arrays depending on whether user provided single
        values or matrices.
        U, V = Synthetic U, V winds (for simulated data only)
        """
        if not hasattr(U, '__len__'):
            self.U = U + 0.0 * self.obs_x
        else:
            if np.shape(U) != np.shape(self.obs_x):
                warnings.warn('U & obs_x/y grid matching issue ...')
                return
            self.U = U
        if not hasattr(V, '__len__'):
            self.V = V + 0.0 * self.obs_y
        else:
            if np.shape(V) != np.shape(self.obs_x):
                warnings.warn('V & obs_x/y grid matching issue ...')
                return
            self.V = V

    def use_input_grid(self, xgrid=None, ygrid=None):
        """
        If real radar data are not used, this method will construct the
        simulated observations grid using user-provided information.
        xgrid = 1-D vector of x locations (km)
        ygrid = 1-D vector of y locations (km)
        """
        if len(xgrid) != len(ygrid):
            print('use_input_grid: x/ygrid sizes don\'t match, failing ...')
            return
        self.obs_x, self.obs_y = np.meshgrid(xgrid, ygrid)
        self.azimuth = np.rad2deg(atan2_array(self.obs_y, self.obs_x))
        self.azimuth = rotate_azimuths(self.azimuth)
        self.slant_range = (self.obs_x**2 + self.obs_y**2)**0.5

    def construct_synthetic_grid(self, az_spacing=2.0, range_spacing=1.0,
                                 max_range=100.0):
        """
        If real radar data are not used, this method will construct the
        simulated observations grid using user-provided information.
        az_spacing = degrees to space out the azimuth data
        range_spacing = km to space out the range data
        max_range = maximum range to consider (km)
        """
        # Azimuth
        if 360 % np.int32(az_spacing) != 0:  # Truncation issue?
            self.az_spacing = 2.0
        else:
            self.az_spacing = az_spacing  # Truncation issue?
        self.range_spacing = range_spacing
        self.number_of_beams = np.int32(360.0/self.az_spacing)
        azimuth = self.az_spacing * np.arange(self.number_of_beams)
        # Range
        self.max_range = np.int32(np.round(max_range))
        if self.max_range % np.int32(range_spacing) != 0:
            self.max_range = 100
            self.range_spacing = 1.0
        self.ngates = np.int32(np.round(np.float(max_range) /
                                        self.range_spacing))
        slant_range = self.range_spacing * np.arange(self.ngates)
        self.azimuth, self.slant_range = np.meshgrid(azimuth, slant_range)
        # Convert to xy coordinates
        self.obs_x = self.slant_range * np.sin(np.deg2rad(self.azimuth))
        self.obs_y = self.slant_range * np.cos(np.deg2rad(self.azimuth))

    def analyze_vad_rings(self, field='VR', sweep_number=0, verbose=False):
        """
        Given a radial velocity sweep, compute VAD on a number of range rings.
        However, only return the U & V median values for all range rings.
        field = Name of Py-ART radial velocity field
        sweep_number = Sweep to consider in analysis
        verbose = Set to True to get debug info
        """
        self.range_rings = 1.0 * np.arange(self.max_range) + 1.0
        self.vad_ws = 0.0 * self.range_rings
        self.vad_wd = 0.0 * self.range_rings
        for i, rng in enumerate(self.range_rings):
            self.vad_ws[i], self.vad_wd[i] = compute_vad_ring(
                self.radar, slant_range=rng, field=field,
                sweep_number=sweep_number, verbose=verbose)
        # Find median U and V from good data
        cond = np.logical_and(self.vad_ws > BAD_DATA_VAL+1,
                              self.vad_wd > BAD_DATA_VAL+1)
        try:
            self.vad_u, self.vad_v = \
                get_u_and_v_from_ws_and_wd(self.vad_ws[cond],
                                           self.vad_wd[cond])
            self.vad_u = np.median(self.vad_u)
            self.vad_v = np.median(self.vad_v)
        except:
            warnings.warn('Not enough data for VAD, returning 0')
            warnings.warn('This means 2DVAR will likely fail ' +
                          'spectacularly as well')
            self.vad_u = 0.0
            self.vad_v = 0.0

    def get_obs_background_field(self, Ub, Vb, grid=None):
        """
        Contains ability to handle 1-D varying wind field. However, recommend
        only supplying scalars as arguments.
        Ub, Vb = Background U & V winds
        grid = 1D array of ranges to consider (from VAD)
        """
        if hasattr(Ub, '__len__') or hasattr(Vb, '__len__'):
            rngf = (self.obs_xf**2 + self.obs_yf**2)**0.5
        if Ub is None:
            self.obs_Ub = 0.0
        elif hasattr(Ub, '__len__'):
            if grid is None:
                self.obs_Ub = 0.0
            else:
                self.obs_Ub = np.interp(rngf, grid, Ub)
        else:
            self.obs_Ub = Ub
        if Vb is None:
            self.obs_Vb = 0.0
        elif hasattr(Vb, '__len__'):
            if grid is None:
                self.obs_Vb = 0.0
            else:
                self.obs_Vb = np.interp(rngf, grid, Vb)
        else:
            self.obs_Vb = Vb
        self.compute_beta_and_m()
        # Following is for Beta as non-radar convention (0 = due East)
        self.obs_vrbf = self.obs_Ub * np.cos(self.obs_Beta) +\
            self.obs_Vb * np.sin(self.obs_Beta)

    def get_analysis_background_field(self, Ub, Vb, grid=None):
        """
        Contains ability to handle 1-D varying wind field. However, recommend
        only supplying scalars as arguments.
        Ub, Vb = Background U & V winds
        grid = 1D array of ranges to consider (from VAD)
        """
        if hasattr(Ub, '__len__') or hasattr(Vb, '__len__'):
            rngf = (self.analysis_xf**2 + self.analysis_yf**2)**0.5
        if Ub is None:
            self.analysis_Ub = 0.0
        elif hasattr(Ub, '__len__'):
            if grid is None:
                self.analysis_Ub = 0.0
            else:
                self.analysis_Ub = np.interp(rngf, grid, Ub)
                self.analysis_Ub = np.reshape(self.analysis_Ub,
                                              (self.N, self.N))
        else:
            self.analysis_Ub = Ub
        if Vb is None:
            self.analysis_Vb = 0.0
        elif hasattr(Vb, '__len__'):
            if grid is None:
                self.analysis_Vb = 0.0
            else:
                self.analysis_Vb = np.interp(rngf, grid, Vb)
                self.analysis_Vb = np.reshape(self.analysis_Vb,
                                              (self.N, self.N))
        else:
            self.analysis_Vb = Vb
        # Following is for Beta as non-radar convention (0 = due East)
        Beta = np.reshape(self.analysis_Beta, (self.N, self.N))
        self.analysis_vrb = self.analysis_Ub * np.cos(Beta) + \
            self.analysis_Vb * np.sin(Beta)
        self.analysis_vtb = -1.0 * self.analysis_Ub * np.sin(Beta) + \
            self.analysis_Vb * np.cos(Beta)

################################


class BaseAnalysis(object):

    def __init__(self, analysis=None):
        if analysis is not None:
            try:
                if not hasattr(analysis, 'analysis_u'):
                    analysis.get_velocity_vectors()
                for var in VAR_LIST:
                    values = getattr(analysis, var)
                    # Mask missing data due to distance filtering
                    if var in ['analysis_vr', 'analysis_vt',
                               'analysis_u', 'analysis_v']:
                        values = np.ma.masked_invalid(values)
                    setattr(self, var, values)
                if hasattr(analysis, 'radar'):
                    self.radar = analysis.radar
                else:
                    self.radar = None
            except:
                warnings.warn('Not a proper analysis object, failing ...')

################################


class AnalysisDisplay(BaseAnalysis):

    def __init__(self, SingleDoppler2D):
        """Requires SingleDoppler2D object as argument"""
        BaseAnalysis.__init__(self, analysis=SingleDoppler2D)

    def plot_velocity_vectors(self, scale=600.0, thin=4, save=None, ax=None,
                              fig=None, xlim=None, ylim=None,
                              axislabels=DEFAULT_LABELS,
                              legend=10.0, title='Vector Velocity Field'):
        """
        One-panel plot of analyzed velocity vectors
        If vectors not yet retrieved will do so before plotting
        scale = scale of arrows
        thin = factor to thin number of arrows by
        save = filename to save to
        ax = axes to use, if any
        fig = figure to use, if any
        xlim, ylim = grid limits (default is analysis grid limits)
        legend = scale for sample velocity vector (m/s)
        axislabels = tuple containing labels for x & y axes
        title = plot title (string)
        """
        ax, fig = self._parse_ax_fig(ax, fig)
        cond = np.logical_and(self.analysis_x % thin == 0,
                              self.analysis_y % thin == 0)
        Q = ax.quiver(self.analysis_x[cond], self.analysis_y[cond],
                      self.analysis_u[cond], self.analysis_v[cond],
                      scale=scale)
        ax.quiverkey(Q, 0.85, 1.02, legend, str(legend)+' m/s',
                     coordinates='axes', labelpos='E')
        self.set_limits(xlim=xlim, ylim=ylim, ax=ax)
        self.label_axes(axislabels=axislabels, ax=ax)
        self.add_title(title, ax=ax)
        self._save_image(save)

    def plot_velocity_contours(self, var='VR', cmap='bwr', mesh_flag=False,
                               levels=DEFAULT_LEVELS, xlim=None, ylim=None,
                               axislabels=DEFAULT_LABELS, colorbar_flag=True,
                               save=None, ax=None, fig=None, title=None):
        """
        Contour plot of either radial or tangential velocities (analyzed)
        cmap = color map
        levels = contour levels
        save = name of image file to save to
        xlim, ylim = limits of plot
        var = Variable to plot ('VR' or 'VT')
        axislabels = tuple comntaining names of axes labels
        colorbar_flag = set to False to suppress colobar
        ax = axes object to use
        fig = figure object to use
        title = title of plot
        mesh_flag = flag to switch between plt.contourf() or plt.pcolormesh()
        """
        var_str, var_to_plot = self._parse_variable(var)
        ax, fig = self._parse_ax_fig(ax, fig)
        if mesh_flag:
            cr = ax.pcolormesh(self.analysis_x, self.analysis_y, var_to_plot,
                               vmin=np.min(levels), vmax=np.max(levels),
                               cmap=cmap)
        else:
            cr = ax.contourf(self.analysis_x, self.analysis_y, var_to_plot,
                             levels=levels, cmap=cmap)
        if title is None or isinstance(title, str) is False:
            title = var_str
        self.add_title(title, ax=ax)
        self.set_limits(xlim=xlim, ylim=ylim, ax=ax)
        self.label_axes(axislabels=axislabels, ax=ax)
        if colorbar_flag:
            plt.colorbar(cr, label='m/s')
        self._save_image(save)

    def plot_radial_tangential_contours(self, cmap='bwr', return_flag=False,
                                        levels=DEFAULT_LEVELS, save=None,
                                        mesh_flag=False):
        """
        Two-panel plot of contours of radial (a) and tangential (b) velocities
        cmap = color map
        return_flag = set to true to retrieve figure/axis objects
        levels = contour levels
        mesh_flag = flag to switch between plt.contourf() or plt.pcolormesh()
        save = name of file to save image to
        """
        fig = plt.figure(figsize=(14, 5))
        ax1 = fig.add_subplot(121)
        plt.suptitle('L = '+str(self.L)+' km',
                     fontsize='large', weight='bold')
        self.plot_velocity_contours(var='VR', cmap=cmap, levels=levels,
                                    mesh_flag=mesh_flag,
                                    title='(a) Radial Velocity')
        ax2 = fig.add_subplot(122)
        self.plot_velocity_contours(var='VT', cmap=cmap, levels=levels,
                                    mesh_flag=mesh_flag,
                                    title='(b) Tangential Velocity')
        self._save_image(save)
        if return_flag:
            return fig, ax1, ax2

    def four_panel_plot(self, scale=600.0, levels=-24.0+4.0*np.arange(13),
                        cmap='bwr', return_flag=False, thin=4, legend=10.0,
                        save=None, name_dz=DEFAULT_DZ, name_vr=DEFAULT_VR,
                        split_cut=False, sweep=0):
        """
        Produces 4-panel plot
        (a) = Observed Vr
        (b) = Reflectivity with analyzed wind vectors overlaid
        (c) = Analyzed radial velocity
        (d) = Analyzed tangential velocity

        semi-exclusive arguments/keywords
        -----------------------------
        return_flag = set to true to return figure and axes objects
        name_dz = name of radar reflectivity field
        name_vr = name of radar Doppler velocity field

        See plot_velocity_vectors() and plot_velocity_contours() for
        more info on arguments and keywords
        """
        self.split_cut = split_cut
        if self.radar is None:
            print('Missing radar object, try again')
            return
        plt.close()
        display = pyart.graph.RadarDisplay(self.radar)
        fig, ax1 = self._four_pan_subplot_a(display, name_vr, levels, cmap,
                                            sweep)
        fig, ax2 = self._four_pan_subplot_b(fig, display, name_dz, scale,
                                            legend, thin, sweep)
        fig, ax3 = self._four_pan_subplot_c(fig, levels, cmap)
        fig, ax4 = self._four_pan_subplot_d(fig, levels, cmap)
        fig, cb1 = self._four_pan_colorbar_1(fig)
        fig, cb2 = self._four_pan_colorbar_2(fig, levels, cmap)
        title_text = self._get_radar_info(sweep)
        fig.suptitle(title_text, fontsize=14, y=0.94)
        self._save_image(save)
        if return_flag:
            return fig, ax1, ax2, ax3, ax4

    def add_title(self, title, ax=None):
        """Adapted from Py-ART"""
        ax = self._parse_ax(ax)
        if isinstance(title, str):
            ax.set_title(title)

    def label_axes(self, ax=None, axislabels=DEFAULT_LABELS):
        """Adapted from Py-ART"""
        ax = self._parse_ax(ax)
        if isinstance(axislabels[0], str):
            ax.set_xlabel(axislabels[0])
        if isinstance(axislabels[1], str):
            ax.set_ylabel(axislabels[1])

    def set_limits(self, xlim=None, ylim=None, ax=None):
        """
        Set the display limits.
        Parameters
        ----------
        xlim: tuple
              optional 2-Tuple containing y-axis limits in km.
              None uses default limits.
        ylim: tuple
              optional 2-Tuple containing x-axis limits in km.
              None uses default limits.
        ax: Axis
            Axis to adjust. None will adjust the current axis.
        Adapted from Py-ART
        """
        ax = self._parse_ax(ax)
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(self.grid_limits)
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(self.grid_limits)

    def _four_pan_subplot_a(self, display, name_vr, levels, cmap, sweep):
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(221)
        if self.split_cut:
            sweep += 1
        display.plot_ppi(name_vr, sweep, vmin=np.min(levels),
                         vmax=np.max(levels), cmap=cmap, colorbar_flag=False,
                         axislabels=DEFAULT_LABELS)
        display.set_limits(xlim=self.grid_limits, ylim=self.grid_limits)
        plt.title('(a) Observed Radial Velocity')
        return fig, ax1

    def _four_pan_subplot_b(self, fig, display, name_dz, scale, legend, thin,
                            sweep):
        ax2 = fig.add_subplot(222)
        display.plot_ppi(name_dz, sweep, vmin=0.0, vmax=65.0, cmap=DZ_CMAP,
                         colorbar_flag=False, axislabels=DEFAULT_LABELS)
        display.set_limits(xlim=self.grid_limits, ylim=self.grid_limits)
        self.plot_velocity_vectors(scale=scale, thin=thin, legend=legend,
                                   title='(b) Vector Velocity Field')
        return fig, ax2

    def _four_pan_subplot_c(self, fig, levels, cmap):
        ax3 = fig.add_subplot(223)
        self.plot_velocity_contours(var='VR', levels=levels, cmap=cmap,
                                    colorbar_flag=False, mesh_flag=True,
                                    title='(c) Analyzed Radial Velocity')
        return fig, ax3

    def _four_pan_subplot_d(self, fig, levels, cmap):
        ax4 = fig.add_subplot(224)
        self.plot_velocity_contours(var='VT', levels=levels, cmap=cmap,
                                    colorbar_flag=False, mesh_flag=True,
                                    title='(d) Analyzed Tangential Velocity')
        return fig, ax4

    def _four_pan_colorbar_1(self, fig):
        a = np.array([[0, 65]])
        fig.add_axes([0.01, 0.01, 0.02, 0.02])
        img1 = plt.imshow(a, cmap=DZ_CMAP)
        plt.gca().set_visible(False)
        cax1 = fig.add_axes([0.92, 0.55, 0.02, 0.35])
        cb1 = plt.colorbar(orientation='vertical', cax=cax1)
        cb1.set_label('dBZ')
        return fig, cb1

    def _four_pan_colorbar_2(self, fig, levels, cmap):
        b = np.array([[np.min(levels), np.max(levels)]])
        fig.add_axes([0.01, 0.01, 0.02, 0.02])
        img2 = plt.imshow(b, cmap=cmap)
        plt.gca().set_visible(False)
        cax2 = fig.add_axes([0.92, 0.125, 0.02, 0.35])
        cb2 = plt.colorbar(cax=cax2, orientation='vertical')
        cb2.set_label('m/s')
        return fig, cb2

    def _get_radar_info(self, sweep):
        """Derived from similar functions in Py-ART"""
        if self.split_cut:
            sweep += 1
        swpstr = '%.1f deg' % self.radar.fixed_angle['data'][0]
        if 'instrument_name' in self.radar.metadata:
            radstr = self.radar.metadata['instrument_name']
        else:
            radstr = ''
        try:
            timstr = self.radar.time['units'][-20:]
        except:
            timstr = ''
        return radstr + ' ' + timstr + ' ' + swpstr

    def _save_image(self, save):
        if save is not None:
            try:
                plt.savefig(save)
            except:
                warnings.warn('Bad name for saving image, try again')

    def _parse_ax(self, ax):
        """Parse and return ax parameter. Adapted from Py-ART."""
        if ax is None:
            ax = plt.gca()
        return ax

    def _parse_ax_fig(self, ax, fig):
        """Parse and return ax and fig parameters. Adapted from Py-ART."""
        if ax is None:
            ax = plt.gca()
        if fig is None:
            fig = plt.gcf()
        return ax, fig

    def _parse_variable(self, var='VR'):
        wstr = 'Don\'t understand. Use var=\'VR\' or \'VT\'. Plotting VR.'
        if isinstance(var, str):
            if var.upper() == 'VR':
                return 'Radial Velocity', self.analysis_vr
            elif var.upper() == 'VT':
                return 'Tangential Velocity', self.analysis_vt
            else:
                warnings.warn(wstr)
                return 'Radial Velocity', self.analysis_vr
        else:
            warnings.warn(wstr)
            return 'Radial Velocity', self.analysis_vr

#############################


class SaveFile(object):

    """
    Purpose of class is create simple I/O interface for
    singledop.AnalysisDisplay
    objects, to avoid repeated 2DVAR computations if the analysis is in good
    shape. Uses pickle to write to or read binary files. Only saves the most
    critical info needed for AnalysisDiaplay methods. Must provide Py-ART
    radar object as argument if analysis came from non-synthetic data.

    Use Examples:
    savefile_instance = singledop.SaveFile(SingleDoppler2D_obj,
                                           filename='example.dat')
        -Saves SingleDoppler2D_obj to './example.dat'

    savefile_instance = singledop.SaveFile('example.dat', radar=radar_obj)
    new_display = singledop.AnalysisDisplay(savefile_instance)
        -Reads from pre-existing './example.dat' file and populates radar
         using pre-existing radar object
        -Creates AnalysisDisplay object
        -Can also set radar='radar_file.nc' to populate radar object from given
         radar file
    """

    def __init__(self, SingleDoppler2D=None, filename=None, filedir='./',
                 radar=None):
        """
        SingleDoppler2D = singledop.SingleDoppler2D object
        filename = Name of file to write to
        filedir = Path to file
        radar = Py-ART radar object or radar file
        """
        if SingleDoppler2D is not None:
            # Account for user just providing string filename for reading as
            # initial argument?
            if isinstance(SingleDoppler2D, str):
                if isinstance(filedir, str):
                    SingleDoppler2D = filedir+SingleDoppler2D
                print('Attempting to read from', SingleDoppler2D)
                self.read_from_file(filename=filedir+SingleDoppler2D,
                                    radar=radar)
                return
            # Rest assumes user provided non-string object
            print('Initializing singledop.SaveFile object')
            self.populate_attributes(SingleDoppler2D)
            if filename is not None and isinstance(filename, str) and \
               isinstance(filedir, str):
                print('Writing to', filedir+filename)
                self.write_to_file(filename, filedir)
        elif filename is not None and isinstance(filename, str) and \
                isinstance(filedir, str):
            print('Reading from', filedir+filename)
            self.read_from_file(filename, filedir, radar)
        else:
            warnings.warn('No valid arguments given, failing ...')

    def populate_attributes(self, SingleDoppler2D):
        """SingleDoppler2D = singledop.SingleDoppler2D object"""
        if hasattr(SingleDoppler2D, 'analysis_vr'):
            self.grid_limits = SingleDoppler2D.grid_limits
            self.analysis_x = SingleDoppler2D.analysis_x
            self.analysis_y = SingleDoppler2D.analysis_y
            self.analysis_vr = SingleDoppler2D.analysis_vr
            self.analysis_vt = SingleDoppler2D.analysis_vt
            if not hasattr(SingleDoppler2D, 'analysis_u'):
                SingleDoppler2D.get_velocity_vectors()
            self.analysis_u = SingleDoppler2D.analysis_u
            self.analysis_v = SingleDoppler2D.analysis_v
            self.L = SingleDoppler2D.L
        else:
            warnings.warn('Not singledop.SingleDoppler2D object, failing ...')
            return

    def write_to_file(self, filename, filedir='./'):
        """
        filename = Name of file to write to
        filedir = Path to file
        """
        filename = filedir + filename
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def read_from_file(self, filename, filedir='./', radar=None):
        """
        filename = Name of file to read from
        filedir = Path to file
        radar = Py-ART radar object or radar file
        """
        filename = filedir + filename
        with open(filename, 'rb') as f:
            loadobj = pickle.load(f)
        self.populate_attributes(loadobj)
        if isinstance(radar, str):
            radar = pyart.io.read(radar)
        self.radar = radar

#############################


class NetcdfSave(object):

    """
    Class to facilitate saving/loading to/from a netCDF file.
    Uses xray module to interface with netCDF.

    Example - Load from netCDF & import into AnalysisDisplay class:
    example = singledop.NetcdfSave('your_file_here.nc')
    display = singledop.AnalysisDisplay(example)
    """
    def __init__(self, analysis=None, filename='singledop_analysis.nc',
                 radar=None):
        """
        analysis = SingleDoppler2D object
        filename = Name of file to write to or load from
        If only passed a string argument with no keyword identifiers, will
        try to load from the string filename.
        """
        self.analysis = analysis
        self.filename = filename
        self.radar = radar
        if self.analysis is not None:
            self.load_or_save()
        else:
            self.load_loop()

    def load_or_save(self):
        if isinstance(self.analysis, str):
            self.filename = self.analysis
            self.load_loop()
        else:
            self.reformat_for_netcdf()
            self.save_to_netcdf()

    def load_loop(self):
        self.load_from_netcdf()
        self.reformat_to_analysis()
        self.check_for_radar_object()

    def reformat_for_netcdf(self):
        """Creates an xray.Dataset object from SingleDoppler2D attributes"""
        if not hasattr(self.analysis, 'analysis_u'):
            self.analysis.get_velocity_vectors()
        if hasattr(self.analysis, 'radar'):
            if self.analysis.radar is not None:
                self.time = self.analysis.radar.time['units'][14:]
                if hasattr(self.analysis.radar.latitude['data'], '__len__'):
                    self.lat = self.analysis.radar.latitude['data'][0]
                    self.lon = self.analysis.radar.longitude['data'][0]
                else:
                    self.lat = self.analysis.radar.latitude['data']
                    self.lon = self.analysis.radar.longitude['data']
                att = {'time': self.time, 'radar_latitude': self.lat,
                       'radar_longitude': self.lon}
            else:
                att = None
        else:
            att = None
        self._get_data_arrays()
        self.ds = xray.Dataset(self.da, attrs=att)

    def save_to_netcdf(self):
        self.ds.to_netcdf(self.filename, format='NETCDF3_CLASSIC')

    def load_from_netcdf(self):
        self.ds = xray.open_dataset(self.filename)

    def check_for_radar_object(self):
        if self.radar is not None:
            if isinstance(self.radar, str):
                self.radar = pyart.io.read(self.radar)

    def reformat_to_analysis(self):
        """
        Creates an analysis object that holds important attributes for
        plotting the results of the single-Doppler retrieval.
        """
        self.analysis = SimpleObject()
        for var in VAR_LIST:
            if var != 'L':
                newatt = np.array(self.ds[var])
            else:
                newatt = np.float(self.ds[var])
            # Sloppy but at least the user can find needed vars anywhere
            setattr(self.analysis, var, newatt)
            setattr(self, var, newatt)

    def _get_data_arrays(self):
        dxy = ['x', 'y']
        km = 'kilometers'
        ms = 'meters per second'
        cow = ' component of wind'
        self.da = {}
        self.da['analysis_x'] = xray.DataArray(
            self.analysis.analysis_x, dims=dxy,
            attrs=var_atts('distance east from radar', km))
        self.da['analysis_y'] = xray.DataArray(
            self.analysis.analysis_y, dims=dxy,
            attrs=var_atts('distance north from radar', km))
        self.da['analysis_u'] = xray.DataArray(
            self.analysis.analysis_u, dims=dxy,
            attrs=var_atts('eastward' + cow, ms))
        self.da['analysis_v'] = xray.DataArray(
            self.analysis.analysis_v, dims=dxy,
            attrs=var_atts('northward' + cow, ms))
        self.da['analysis_vr'] = xray.DataArray(
            self.analysis.analysis_vr, dims=dxy,
            attrs=var_atts('radial' + cow, ms))
        self.da['analysis_vt'] = xray.DataArray(
            self.analysis.analysis_vt, dims=dxy,
            attrs=var_atts('tangential' + cow, ms))
        self.da['grid_limits'] = xray.DataArray(
            self.analysis.grid_limits,
            attrs=var_atts('x & y boundaries for rectangular grid', km))
        self.da['L'] = xray.DataArray(
            self.analysis.L, attrs=var_atts('decorrelation length scale', km))

#############################


class SimpleObject(object):
    pass

#############################

# More classes go here

##############################
# Independent functions follow
##############################


def compute_vad_ring(radar, field='VR', slant_range=30.0, sweep_number=0,
                     verbose=False):
    """
    Major reference
    ---------------
    Browning, K. A., and R. Wexler, 1968: The determination of kinematic
    properties of a wind field using Doppler radar. J. Appl. Meteorol., 7,
    105-113.

    Given arguments, return wind speed and direction via VAD analysis on single
    ring of radar data. Returns NoneType if not enough data. Radial velocity
    should be filtered for non-met echo and unfolded b4 ingest. You will need
    to iterate over multiple rings to examine a full volume (or sweep). Assumes
    PPI scanning style and data arrangement.

    Arguments
    ---------
    radar = Py-ART radar object
    field = string name of radial velocity field
    slant_range = range ring (km) to consider
    sweep_number = sweep to consider
    verbose = set to True to get warnings if not enough data for VAD
    """
    radar_sweep = radar.extract_sweeps([sweep_number])
    rindex = np.int32(np.round((slant_range * RNG_MULT -
                                radar_sweep.range['data'][0]) /
                               (radar_sweep.range['data'][1] -
                                radar_sweep.range['data'][0])))
    # Sometimes the '_FillValue' field is missing, deal with this gracefully
    if '_FillValue' not in radar_sweep.fields[field]:
        radar_sweep.fields[field]['_FillValue'] = BAD_DATA_VAL
    # What if field is not a masked array?
    try:
        vr = radar_sweep.fields[field]['data'][:, rindex].filled(
            fill_value=radar_sweep.fields[field]['_FillValue'])
    except AttributeError:
        vr = radar_sweep.fields[field]['data'][:, rindex]
    cond = vr != radar_sweep.fields[field]['_FillValue']
    vr = vr[cond]
    az = radar_sweep.azimuth['data'][cond]
    if np.size(vr) > 1:
        frac = 2.0 / float(np.size(vr))
        a1 = frac * np.sum(vr * np.cos(np.deg2rad(az)))
        b1 = frac * np.sum(vr * np.sin(np.deg2rad(az)))
        vh = (a1**2 + b1**2)**0.5 /\
            np.cos(np.deg2rad(radar_sweep.fixed_angle['data'][0]))
        if b1 < 0.0:
            phi = np.pi/2.0 - math.atan(a1/b1)
        else:
            phi = 3.0*np.pi/2.0 - math.atan(a1/b1)
        return vh, np.rad2deg(phi)
    else:
        if verbose:
            warnings.warn('Not enough data, failing')
        return BAD_DATA_VAL, BAD_DATA_VAL


def rotate_azimuths(azimuth):
    """
    Adjusts azimuths returned from atan2 to conform to radar convention
    (i.e., 0-deg = North)
    Expects array input
    """
    new_azimuth = 0.0 * azimuth
    cond = np.logical_and(azimuth >= -180, azimuth < 90)
    new_azimuth[cond] = 90.0 - azimuth[cond]
    cond = np.logical_and(azimuth >= 90, azimuth <= 180)
    new_azimuth[cond] = 360.0 + (90.0 - azimuth[cond])
    return new_azimuth


def atan2_array(y, x):
    """
    Expects arrays due to Boolean conditions
    Currently returns angles in non-radar convention
    Change?
    """
    if not hasattr(y, '__len__') or not hasattr(x, '__len__'):
        warnings.warn('Arguments need to be arrays, failing ...')
        return None
    atan2 = 0.0 * y  # At x,y=0 atan2=0 to avoid numerical issues
    cond = x > 0
    atan2[cond] = np.arctan(y[cond]/x[cond])
    cond = np.logical_and(x < 0, y >= 0)
    atan2[cond] = np.arctan(y[cond]/x[cond]) + np.pi
    cond = np.logical_and(x < 0, y < 0)
    atan2[cond] = np.arctan(y[cond]/x[cond]) - np.pi
    cond = np.logical_and(x == 0, y > 0)
    atan2[cond] = np.pi/2.0
    cond = np.logical_and(x == 0, y < 0)
    atan2[cond] = -1.0 * np.pi / 2.0
    return atan2


def get_r(x, y, x1, y1):
    """
    Get r vector following Xu et al. (2006) Eq. 4.2
    x, y = arrays; x1, y1 = single points; or vice-versa
    """
    return ((x-x1)**2 + (y-y1)**2)**0.5


def vector_mask(x, y, x1, y1, L):
    """
    Get masked velocity field following Xu et al. (2006) Eq. 4.2
    x, y = arrays; x1, y1 = single points; or vice-versa
    L is decorrelation length scale
    """
    return np.exp(-1.0 * (get_r(x, y, x1, y1))**2 / (2.0 * L**2))


def get_Crr(x, y, x1, y1, Beta, Beta1, L, sigma):
    """
    Get auto-correlation matrix components following Xu et al. (2006) Eq. 4.2
    Assumes a=0, b=1
    x, y = arrays; x1, y1 = single points; or vice-versa
    L is decorrelation length scale
    Beta = array of angles (radians); Beta1 = single angle; or vice-versa
    """
    return (sigma**2) * vector_mask(x, y, x1, y1, L) * np.cos(Beta1 - Beta)


def get_Crt(x, y, x1, y1, Beta, Beta1, L, sigma):
    """
    Get cross-correlation matrix components following Xu et al. (2006) Eq. 4.2
    Assumes a=0, b=1
    x, y = arrays; x1, y1 = single points; or vice-versa
    L is decorrelation length scale
    Beta = array of angles (radians); Beta1 = single angle; or vice-versa
    """
    return (sigma**2) * vector_mask(x, y, x1, y1, L) * np.sin(Beta1 - Beta)


def get_x_and_y_from_radar(radar, sweep_number):
    """Input radar object and sweep number, return xy from radar (km, 2D)"""
    azimuth_sweep = get_sweep_azimuths(radar, sweep_number)
    elev_sweep = get_sweep_elevations(radar, sweep_number)
    sr_1d = radar.range['data'][:] / RNG_MULT
    sr_2d, az_2d = np.meshgrid(sr_1d, azimuth_sweep)
    el_2d = np.meshgrid(sr_1d, elev_sweep)[1]
    xx, yy, zz = radar_coords_to_cart(sr_2d, az_2d, el_2d)
    xx /= RNG_MULT
    yy /= RNG_MULT
    return xx, yy


def get_u_and_v_from_ws_and_wd(ws, wd, radar_convention=True, radians=False):
    """
    ws = wind speed (units irrelevant, but user should keep track of this)
    wd = wind direction (deg)
    radar_convention = True means wd is deg from N (+ = CW rotation),
                       False means wd is deg from E (+ = CCW rotation)
    radians = set to True if wd is provided in radians (False = deg)
    """
    if not radians:
        wd = np.deg2rad(wd)
    if radar_convention:
        return -1.0*ws*np.sin(wd), -1.0*ws*np.cos(wd)  # U, V
    else:
        return ws*np.cos(wd), ws*np.sin(wd)  # U, V


def var_atts(name, units):
    return {'name': name, 'units': units}

##############################
# Internal functions follow
##############################
