'''
This Python module is intended to provide all mathematical functions for Evaluix \n
It is written blockwise for maintenance:
    1. Import all required external modules (numpy, pandas, scipy, typing and lmfit)
    2. Basic functions
    3. Data manipulation functions
    4. Evaluation functions

'''
###############################################################################        
# 1. Import necessary modules
###############################################################################

import warnings

import numpy as np
import pandas as pd
from lmfit import Model, Parameters
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.integrate import quad
from scipy.optimize import curve_fit, minimize
from scipy.signal import find_peaks, peak_widths, savgol_filter

# import support functions
from .support_functions import _check_array_like, _safe_stderr

SAT_COND = 0.95 # Condition for saturation: 95% of the maximum value is considered as saturation whereby this value can be changed as argument
MAD_TO_SIGMA = 1.4826  # conversion factor from median absolute deviation to standard deviation

#%%
###############################################################################        
# 2. Basic (Fit) Functions
###############################################################################
def linear(xdata: float | list | pd.DataFrame | pd.Series | np.ndarray, a: float, b: float):
    """
    Linear model for simple trends.

    Model
    -----
    f(x) = a x + b

    Interpretation
    --------------
    This function is just a numerical model. In the hysteresis analysis here,
    it is mostly used to describe approximately linear contributions in saturation. 
    For example dia- and paramagnetic contributions to magnetization curves, or a 
    branch-wise hysteresis opening at saturation due to instrumental artifacts.

    Notes
    -----
    - Should only be used over ranges where a linear approximation is justified.

    Parameters
    ----------
    xdata : float | list | pd.DataFrame | pd.Series | np.ndarray
        Input value(s) (typically named ``x`` in fit functions).
    a : float
        Slope parameter.
    b : float
        Constant offset parameter.

    Returns
    -------
    y_data : numpy.ndarray
        Modeled value(s) of the linear function.

    Raises
    ------
    TypeError
        If `xdata` is not of a supported type.

    Examples
    --------
    >>> linear([0, 1, 2], 2.0, -1.0)
    array([-1.,  1.,  3.])
    >>> linear([-3, 0, 3], 0.5, 0.0)
    array([-1.5,  0. ,  1.5])
    """
    # check xdata format
    _check_array_like(xdata, 'xdata')

    x_data = np.asarray(xdata).copy()
    return a * x_data + b

def quadratic(xdata: float | list | pd.DataFrame | pd.Series | np.ndarray, a: float, b: float, c: float):
    """
    Quadratic model for smooth trends.

    Model
    -----
    f(x) = a x^2 + b x + c

    Parameters
    ----------
    xdata : float | list | pd.DataFrame | pd.Series | np.ndarray
        Input value(s) (typically named x in fit functions).
    a : float
        Quadratic coefficient.
    b : float
        Linear coefficient.
    c : float
        Constant offset.

    Returns
    -------
    y_data : numpy.ndarray
        Modeled value(s) of the quadratic function.
        
    Raises
        ------
        TypeError
            If `xdata` is not of a supported type.
    """
    _check_array_like(xdata, 'xdata')

    x_data = np.asarray(xdata).copy()
    return a * x_data**2 + b * x_data + c

def polynomial(xdata: float | list | pd.DataFrame | pd.Series | np.ndarray, *args: float | list):
    """
    Polynomial model for complexer trends.

    Model
    -----
    f(x) = a_0 x^n + a_1 x^(n-1) + ... + a_n

    The order is determined by the number of coefficients in ``args``.
    For example, 3 coefficients correspond to a quadratic model.

    Interpretation
    --------------
    This function is just a numerical model. In the hysteresis analysis here,
    it is mostly used to describe approximately higher order contributions in 
    saturation. For example dia- and paramagnetic contributions to magnetization 
    curves, or a branch-wise hysteresis opening at saturation due to instrumental 
    artifacts.

    Notes
    -----
    - Coefficients are interpreted in descending powers of ``x``.
    - To skip a specific power, set that coefficient to ``0``.
    - High polynomial orders may overfit (noisy) data and produce unstable behavior.

    Parameters
    ----------
    xdata : float | list | pd.DataFrame | pd.Series | np.ndarray
        Input value(s) (typically named x in functions).
    *args : float | list
        Polynomial coefficients in descending powers.
        For ``m`` coefficients, the model order is ``m-1``.
        Example:
        ``args=(2, -3, 1)`` gives ``f(x)=2x^2-3x+1``.

    Returns
    -------
    y_data : numpy.ndarray
        Modeled value(s) of the polynomial function.

    Raises
    ------
    TypeError
        If `xdata` is not of a supported type.

    Examples
    --------
    >>> polynomial([1, 2, 3], 1, 0, -1)
    array([0., 3., 8.])
    >>> polynomial([0, 1, 2], 2, -3, 1)  # 2x^2 - 3x + 1
    array([1., 0., 3.])
    """
    # check xdata format
    _check_array_like(xdata, 'xdata')

    x_data = np.asarray(xdata).copy()
    y_data = np.zeros_like(x_data, dtype=float)

    for i, coef in enumerate(args):
        y_data = y_data + coef * x_data ** (len(args) - 1 - i)

    return y_data

def arctan(
    xdata: float | list | pd.DataFrame | pd.Series | np.ndarray, 
    a: float, 
    b: float, 
    c: float, 
    d: float,
    e: float,
):
    """
    Single-branch arctan model for hysteresis-like magnetization curves M(H).

    Model
    -----
    M(H) = a + (2 b / pi) * arctan[c * (H - d + e)]

    This phenomenological function is used to describe one branch of a
    major hysteresis loop. For a full loop, the sign of `e` is inverted
    between decreasing and increasing field branches.
    
    Physics Interpretation
    ----------------------
    - `H` (`xdata`) is the applied magnetic field.
    - `M` is the measured magnetic response (e.g., normalized magnetization M/Ms, 
    Kerr intensity, magnetic moment, or other signals).
    - `a` is a vertical offset (background/bias signal).
    - `b` sets the saturation amplitude/magnetization.
    - `c` controls transition sharpness around switching (larger `c` -> steeper
    reversal).
    - `d` is a common horizontal shift (often an exchange-bias field shift).
    - `e` is a branch-dependent field offset (coercive field shift). Use `+e` for
    the increasing field branch and `-e` for the decreasing field branch.

    Notes
    -----
    - This is an empirical fit function. It captures loop shapes robustly but may 
    not cover all physical mechanisms.
    - Ensure consistent units: `xdata`, `d`, and `e` must share field units;
    `c` has inverse field units.

    Parameters
    ----------
    xdata : float | list | pd.DataFrame | pd.Series | np.ndarray
        Applied field values.
    a : float
        Vertical offset of the signal.
    b : float
        Amplitude scale of the branch, i.e. saturation magnetization.
    c : float
        Steepness parameter (inverse field scale).
    d : float
        Global horizontal shift of the branch, i.e. exchange bias or field offset.
    e : float
        Branch-dependent horizontal offset (changes sign between branches).

    Returns
    -------
    y_data : numpy.ndarray
        Modeled signal values for the provided `xdata` fields.

    Raises
    ------
    TypeError
        If `xdata` is not of a supported type.

    Examples
    --------
    >>> arctan([-10, 0, 10], a=0.0, b=1.0, c=0.1, d=0.0, e=4.0)
    array([-0.344...,  0.242...,  0.605...])
    """
    # check xdata format
    _check_array_like(xdata, 'xdata')

    x_data = np.asarray(xdata)
    return a + b * 2/np.pi * np.arctan(c * (x_data - d + e))

def arctan_hys(
    xdata: float | list | pd.DataFrame | pd.Series | np.ndarray,
    a: float,
    b_1: float,
    c_1: float,
    d_1: float,
    e_1: float,
):
    """
    Two-branch arctan hysteresis model for a single loop.

    Model
    -----
    The function combines two branch models with opposite signs of ``e``:

    - Increasing-field branch:
      ``M_up(H) = a + (2 b / pi) * arctan[c * (H - d + e)]``
    - Decreasing-field branch:
      ``M_down(H) = a + (2 b / pi) * arctan[c * (H - d - e)]``

    Interpretation
    --------------
    This is a phenomenological major loop representation based on the
    single-branch `arctan` model.

    - ``a``: vertical offset (background/bias).
    - ``b``: saturation amplitude/magnetization.
    - ``c``: switching steepness (inverse field scale).
    - ``d``: common horizontal shift (i.e. exchange bias field).
    - ``e``: branch-dependent offset (i.e. coercive field).

    Notes
    -----
    - For scalar ``xdata``, both branches are evaluated at the same field value.
    - For array-like ``xdata``, the sequence is split into two halves:
      first half uses ``+e``, second half uses ``-e``.
    - If the number of points is odd, the center point is included in both branches 
      (i.e., duplicated) to maintain symmetry.
    - For a positive ``e``, the first half of the data corresponds to the increasing-field 
      branch, and the second half to the decreasing-field branch. This convention is 
      reversed by using a negative ``e``.

    Parameters
    ----------
    xdata : float | list | pd.DataFrame | pd.Series | np.ndarray
        Field value(s). Scalar input evaluates both branches at one point;
        array-like input is split into two branch segments.
    a : float
        Vertical offset of the signal.
    b : float
        Amplitude scale of the loop, i.e. saturation magnetization.
    c : float
        Steepness parameter (inverse field scale).
    d : float
        Global horizontal shift of the loop, i.e. exchange bias or field offset.
    e : float
        Branch-dependent horizontal offset (changes sign between branches).

    Returns
    -------
    y_data : tuple or numpy.ndarray
        If ``xdata`` is scalar, returns ``(mean_abs, ydata1, ydata2)`` where
        ``mean_abs`` is the mean of the absolute branch values.
        If ``xdata`` is array-like, returns concatenated branch values.
        
    Raises
    ------
    TypeError
        If `xdata` is not of a supported type.

    Examples
    --------
    >>> tan_hys(1.0, 0.0, 1.0, 0.2, 2.0, 5.0)
    (0.493..., 0.429..., -0.557...)
    >>> tan_hys([-6, -3, 0, 0, 3, 6], 0.0, 1.0, 0.3, 0.0, 2.0)
    array([-0.557..., -0.185..., 0.344..., -0.344..., 0.185..., 0.557...])
    """
    _check_array_like(xdata, 'xdata')
    
    # if arctan of a single value is wanted. Return the mean of both branches
    # as well as the individual branches
    if isinstance(xdata, float): # for calculating single values
        ydata1 = arctan(xdata, a, b_1, c_1, d_1, e_1)
        ydata2 = arctan(xdata, a, b_1, c_1, d_1, -e_1)
        return np.mean([np.abs(ydata1), np.abs(ydata2)]), ydata1, ydata2
    
    elif isinstance(xdata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        #if xdata is given as a list (hysteresis), split it correspondingly into two branches
        x_data = np.asarray(xdata)

        # This check doesnt make sense as this is a fit function without knowledge of ydata.
        # # Check if the length of xdata is odd, i.e.the center point contributes to both branches.
        # # Duplicate the center point in this case so that both branches are equally long.
        # if len(xdata) % 2 != 0:
        #     center_index = len(xdata) // 2
        #     xdata = np.insert(xdata, center_index, xdata[center_index])

        # Split the array into two halves using slicing
        mid_index = len(x_data) // 2
        Xdata1 = x_data[:mid_index]
        Xdata2 = x_data[mid_index:]

        ydata1 = arctan(Xdata1, a, b_1, c_1, d_1, e_1)
        ydata2 = arctan(Xdata2, a, b_1, c_1, d_1, -e_1)
        return np.append(ydata1, ydata2)

def double_arctan_hys(
    xdata: float | list | pd.DataFrame | pd.Series | np.ndarray,
    a: float,
    b_1: float,
    c_1: float,
    d_1: float,
    e_1: float,
    b_2: float,
    c_2: float,
    d_2: float,
    e_2: float,
):
    """    
    Two arctan hysteresis models for a hysteresis curve with two loops.

    Model
    -----
    The function combines two arctan_hys models with their own set of parameters 
    ``b, c, d & e``:

    - Increasing-field branch:
      ``M_up(H) = arctan(a, b_1, c_1, d_1, e_1) + arctan(0, b_2, c_2, d_2, e_2)``
    - Decreasing-field branch:
      ``M_down(H) = arctan(a, b_1, c_1, d_1, -e_1) + arctan(0, b_2, c_2, d_2, -e_2)``

    Interpretation
    --------------
    This is a phenomenological major loop representation based on the single-branch 
    `arctan` model. Four branches are calculated in pairs of two, sharing the parameters
    `b, c, d & +-e`. I.e. two combined major loops.

    - ``a``: vertical offset (background/bias).
    - ``b``: saturation amplitude/magnetization.
    - ``c``: switching steepness (inverse field scale).
    - ``d``: common horizontal shift (i.e. exchange bias field).
    - ``e``: branch-dependent offset (i.e. coercive field).

    Notes
    -----
    - For scalar ``xdata``, both branches are evaluated at the same field value.
    - For array-like ``xdata``, the sequence is split into two halves:
      first half uses ``+e_1 & +e_2``, second half uses ``-e_1 & -e_2``.
    - If the number of points is odd, the center point is included in both branches 
      (i.e., duplicated) to maintain symmetry.
    - For positive ``e`` values, the first half of the data corresponds to the increasing-field 
      branch, and the second half to the decreasing-field branch. This convention is 
      reversed by using negative ``e`` values. The author is unaware of physical cases in 
      which mixed signs make sense, but the function can handle this as well.
    
    Parameters
    ----------
    xdata : float | list | pd.DataFrame | pd.Series | np.ndarray
        Field value(s). Scalar input evaluates both branches at one point;
        array-like input is split into two branch segments.
    a : float
        Vertical offset of the signal. It is only applied ONCE.
    b_1 : float
        Amplitude scale of the first loop, i.e. saturation magnetization.
    c_1 : float
        Steepness parameter of the first loop (inverse field scale).
    d_1 : float
        Global horizontal shift of the first loop, i.e. exchange bias or field offset.
    e_1 : float
        Branch-dependent horizontal offset of the first loop (changes sign between branches).
    b_2 : float
        Amplitude scale of the second loop, i.e. saturation magnetization.
    c_2 : float
        Steepness parameter of the second loop (inverse field scale).
    d_2 : float
        Global horizontal shift of the second loop, i.e. exchange bias or field offset.
    e_2 : float
        Branch-dependent horizontal offset of the second loop (changes sign between branches).

    Returns
    -------
    y_data : tuple or numpy.ndarray
        If xdata is a single value, returns a tuple of (mean, ydata1, ydata2).
        If xdata is a list or array, returns the concatenated ydata for both branches.
        
    Raises
    ------
    TypeError
        If `xdata` is not of a supported type.

    Examples
    --------
    >>> double_tan_hys(1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) # for single values
    (mean_value, ydata1, ydata2)
    >>> double_tan_hys([1, 2, 3, 4, 5, 6], 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) # for lists
    array([ydata1_values, ydata2_values])
    """
    _check_array_like(xdata, 'xdata')

    # if double arctan of a single value is wanted. Return the mean of both branches
    # as well as the individual branches
    if isinstance(xdata, float): # for calculating single values
        ydata1 =  arctan(xdata, a, b_1, c_1, d_1, e_1) + arctan(xdata, 0, b_2, c_2, d_2, e_2)
        ydata2 =  arctan(xdata, a, b_1, c_1, d_1, -e_1) + arctan(xdata, 0, b_2, c_2, d_2, -e_2)
        return np.mean([np.abs(ydata1), np.abs(ydata2)]), ydata1, ydata2
    
    elif isinstance(xdata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        # forward to tan_hys() function and its logic
        return arctan_hys(xdata, a, b_1, c_1, d_1, e_1) + arctan_hys(xdata, 0, b_2, c_2, d_2, e_2)

def mult_arctan_hys(
    xdata: float | list | pd.DataFrame | pd.Series | np.ndarray,
    **kwargs: dict
    ):

    """
    Multi-loop arctan hysteresis model with an arbitrary number of components (loops).

    Model
    -----
    The function superposes ``n`` arctan hysteresis components:

    - Increasing-field branch:
      ``M_up(H) = a + sum_{i=1..n} (2 b_i / pi) * arctan[c_i * (H - d_i + e_i)]``
    - Decreasing-field branch:
      ``M_down(H) = a + sum_{i=1..n} (2 b_i / pi) * arctan[c_i * (H - d_i - e_i)]``

    The number of loops ``n`` is taken from the amount of provided keyword parameters.
    since each loop requires 4 parameters (b, c, d, e) plus one global offset a the 
    number of loops is ``n = (len(kwargs) - 1) / 4`` 

    Interpretation
    --------------
    This is a phenomenological extension of `arctan_hys` / `double_arctan_hys` for
    complex loops that cannot be represented by only one or two switching components.

    - ``a``: global vertical offset (shared by all components).
    - ``b_i``: amplitude of component ``i``.
    - ``c_i``: steepness of component ``i`` (inverse field scale).
    - ``d_i``: common horizontal shift (exchange-bias shift or field offset) of component ``i``.
    - ``e_i``: branch-dependent offset (coercive-field shift) of component ``i``.

    Notes
    -----
    - Parameter names must follow: ``a, b_1, c_1, d_1, e_1, ..., b_n, c_n, d_n, e_n``.
    - For scalar ``xdata``, both branches are evaluated at the same field value.
    - For array-like ``xdata``, branch logic is delegated to `arctan_hys` and follows
      the same split convention as in that function.
    - Increasing model complexity (larger ``n``) can improve fit flexibility but also
      increases risk of parameter correlation and overfitting.
    - with ``n=1`` or ``n=2`` this functions is equal to `arctan_hys` or 
      `double_arctan_hys`, respectively.

    Parameters
    ----------
    xdata : float | list | pd.DataFrame | pd.Series | np.ndarray
        Field value(s). Scalar input evaluates both branches at one point;
        array-like input is split into branch segments by `arctan_hys`.
    kwargs : dict
        Dictionary of model parameters.
        Required format is ``4*n + 1`` entries with keys:
        ``a, b_1, c_1, d_1, e_1, ..., b_n, c_n, d_n, e_n``.

    Returns
    -------
    y_data : tuple or numpy.ndarray
        If ``xdata`` is scalar, returns ``(mean_abs, ydata1, ydata2)`` where
        ``mean_abs`` is the mean of the absolute branch values.
        If ``xdata`` is array-like, returns concatenated branch values.

    Raises
    ------
    TypeError
        If ``kwargs`` does not follow the required naming/length convention or
        contains values of unsupported types.
    TypeError
        If ``xdata`` is not of a supported type.

    Examples
    --------
    >>> mult_arctan_hys(0.0, a=0.0, b_1=1.0, c_1=0.2, d_1=0.0, e_1=4.0)
    (0.429.., 0.429..., -0.429...)
    >>> mult_arctan_hys(
    ...     [-6, -3, 0, 0, 3, 6],
    ...     a=0.0,
    ...     b_1=0.8, c_1=0.25, d_1=-1.0, e_1=2.5,
    ...     b_2=0.5, c_2=0.40, d_2=1.2, e_2=1.0,
    ... )
    array([-0.662..., -0.225...,  0.340..., -0.412...,  0.281..., 0.744...])
    """
    # check len of list to determine number of arctan functions
    if len(kwargs) % 4 != 1 or len(kwargs) < 5:
        raise TypeError(f'Length of kwargs dict must be 4*n + 1, where n >= 1 is the number of arctan functions. Currently, len(kwargs) = {len(kwargs)}.')

    _check_array_like(xdata, 'xdata')

    # Check if all key-value-pairs have the correct style:
    for key, values in kwargs.items():
        if not isinstance(values, float):
            raise TypeError(f'All values in kwargs dict must be float. Currently, {key} has value {values} of type {type(values)}.')
        if not isinstance(key, str):
            raise TypeError(f'All keys in kwargs dict must be strings. Currently, {key} is of type {type(key)}.')

        # Check if key == 'a' or key starts with 'b', 'c', 'd', or 'e' and is followed by a number
        if key != 'a' and not (key[0] in ['b', 'c', 'd', 'e'] and key.split('_')[-1].isdigit()):
            raise TypeError(f'All keys in kwargs dict must be "a" or start with "b", "c", "d", or "e" followed by a number. Currently, {key} does not follow this style.')

    n_arctan = (len(kwargs) - 1) // 4

    # Check for naming, i.e. if all n_arctan have b,c,d and e
    for n in range(1, n_arctan+1):
        keys = [f"b_{n}", f"c_{n}", f"d_{n}", f"e_{n}"]
        for key in keys:
            if key not in kwargs:
                raise TypeError(f"The parameter {key} is not provided for hysteresis loop {n}.")

    if isinstance(xdata, float): # for calculating single values
        ydata1 = arctan(xdata, kwargs['a'], kwargs['b_1'], kwargs['c_1'], kwargs['d_1'], kwargs['e_1'])
        ydata2 = arctan(xdata, kwargs['a'], kwargs['b_1'], kwargs['c_1'], kwargs['d_1'], -kwargs['e_1'])
        if n_arctan > 1:
            for n in range(1, n_arctan):
                b = kwargs['b_' + str(n+1)]
                c = kwargs['c_' + str(n+1)]
                d = kwargs['d_' + str(n+1)]
                e = kwargs['e_' + str(n+1)]
                ydata1 += arctan(xdata, 0, b, c, d, e)
                ydata2 += arctan(xdata, 0, b, c, d, -e)
        return np.mean([np.abs(ydata1), np.abs(ydata2)]), ydata1, ydata2

    elif isinstance(xdata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        y_data = arctan_hys(xdata, kwargs['a'], kwargs['b_1'], kwargs['c_1'], kwargs['d_1'], kwargs['e_1'])
        if n_arctan > 1:
            for n in range(1, n_arctan):
                b = kwargs['b_' + str(n+1)]
                c = kwargs['c_' + str(n+1)]
                d = kwargs['d_' + str(n+1)]
                e = kwargs['e_' + str(n+1)]
                y_data += arctan_hys(xdata, 0, b, c, d, e)

        return y_data

#%%
###############################################################################        
# 3. Data Manipulation
###############################################################################

def invert_axis(
        xdata: float | list | pd.DataFrame | pd.Series | np.ndarray,
        ydata: float | list | pd.DataFrame | pd.Series | np.ndarray,
        axis: str = 'x'):
    """
    Invert the sign of ``xdata`` and/or ``ydata`` for axis-convention correction.

    This utility is primarily used when measurement sign conventions were set
    inconsistently during experiments, for example in Kerr microscopy where
    contrast polarity and field direction can be user-dependent.

    Operation
    ---------
    - ``axis='x'``: returns ``(-xdata, ydata)``
    - ``axis='y'``: returns ``(xdata, -ydata)``
    - ``axis='both'``: returns ``(-xdata, -ydata)``

    Choosing ``'both'`` is equivalent to a 180 degree rotation of the curve
    around the origin in the ``(x, y)`` plane.

    Parameters
    ----------
    xdata : float | list | pd.DataFrame | pd.Series | np.ndarray
        Independent variable data (e.g., applied field).
    ydata : float | list | pd.DataFrame | pd.Series | np.ndarray
        Dependent variable data (e.g., magnetization, Kerr signal).
    axis : str, optional
        Axis selection for sign inversion. Must be one of ``'x'``, ``'y'``,
        or ``'both'``. Default is ``'x'``.

    Returns
    -------
    tuple[float | list | pd.DataFrame | pd.Series | np.ndarray, float | list | pd.DataFrame | pd.Series | np.ndarray]
        Sign-corrected ``(x_data, y_data)`` according to ``axis``.
        
    Raises
    ------
    TypeError
        If ``axis`` is not one of ``'x'``, ``'y'``, or ``'both'``.
        If ``xdata`` or ``ydata`` is not of a supported type.

    Examples
    --------
    >>> invert_axis(pd.DataFrame([1, 2, 3]), pd.DataFrame([4, 5, 6]), axis='x')
    (   0\n0 -1\n1 -2\n2 -3,    0\n0  4\n1  5\n2  6)
    >>> invert_axis([1, 2], [-3, -4], axis='both')
    ([-1, -2], [3, 4])
    """
    _check_array_like(xdata, 'xdata')
    _check_array_like(ydata, 'ydata')

    if axis == 'x':
        return -1 * xdata, ydata
    elif axis == 'y':
        return xdata, -1 * ydata
    elif axis == 'both':
        return -1 * xdata, -1 * ydata
    else:
        raise TypeError(f'Axis not recognized. Please choose between x, y, or both, not {axis}')


def del_outliers(
    ydata: list | pd.DataFrame | pd.Series | np.ndarray,
    threshold: float = 5.0,
    neighbours: int = 10,
    return_details: bool = False,
 ):
    """
    Remove outliers from the dataset based on a specified threshold and number of neighbours.

    An outlier is defined as a point that differs from the mean of its locally surrounding data points 
    by more than `threshold` times the mean absolute difference (MAD). If a point is identified as an 
    outlier, it is replaced by the mean of its `neighbours`.

    Parameters
    ----------
    ydata : list | pd.DataFrame | pd.Series | np.ndarray
        Output value(s) of the dataset (typically named y in functions).
    threshold : float, optional
        Threshold for the difference of a point to the mean of all points. The default is 5.
    neighbours : int, optional
        Number of neighbours to be taken into account for the second mean calculation as well as 
        for the linear interpolation. The default is 10.
    return_details : bool, optional
        If True, also return details about the outlier detection process. The default is False.

    Returns
    -------
    y_data : numpy.ndarray
        ydata with outliers removed and replaced by the mean of their neighbours.
    numpy.ndarray, optional
        If `return_details` is True, also returns details about the outlier detection process.

    Raises
    ------
    TypeError
        If `ydata` is not of a supported type.

    Examples
    --------
    >>> del_outliers([1, 2, 3, 100, 5, 6, 7])
    array([1, 2, 3, 4, 5, 6, 7])

    """
    # Validate input type
    _check_array_like(ydata, 'ydata')

    # Convert to numpy array if it's a pandas Series or DataFrame
    y_data = np.asarray(ydata).copy()
    n = len(y_data)

    # Handle empty input
    if n == 0:
        if return_details:
            details = {
                "outliers": np.array([], dtype=bool),
                "indices_outliers": np.array([], dtype=int),
            }
            return y_data, details
        return y_data

    # Local median and median absolute deviation (MAD) for outlier detection
    # MAD = median of the absolute deviations from the median, i.e. a measure for how much the data deviates.
    # The median and MAD are quite robust against single outliers
    med_neigh = np.zeros_like(y_data)
    sigma_neigh = np.zeros_like(y_data)

    for i in range(n):
        # handle edge cases by adjusting the window of neighbours
        start = max(0, i - neighbours)
        end = min(n, i + neighbours + 1)
        neigh_slice = y_data[start:end]

        # calculate median and MAD for the current window of neighbours
        med_i = np.median(neigh_slice)
        mad_i = np.median(np.abs(neigh_slice - med_i))

        # save median and MAD for the current point, convert MAD to standard deviation equivalent using the constant MAD_TO_SIGMA for normal distribution
        # See https://en.wikipedia.org/wiki/Median_absolute_deviation
        med_neigh[i] = med_i
        sigma_neigh[i] = MAD_TO_SIGMA * mad_i if mad_i > 0 else 1e-6  # Avoid division by zero

    # Calculate local outlier score (deviation from local median, relative to local MAD)
    local_score = np.abs(y_data - med_neigh) / sigma_neigh

    # Check if local score exceeds the threshold to identify outliers
    outliers = local_score > threshold
    outlier_indices = np.where(outliers)[0]

    # Replace outliers with local interpolation (mean of neighbours excluding the outlier)
    for outlier in outlier_indices:
        if outlier:
            start = max(0, outlier - neighbours)
            end = min(outlier + neighbours + 1, n)
            neigh_slice = np.concatenate([y_data[start:outlier], y_data[outlier + 1:end]])
            if len(neigh_slice):
                y_data[outlier] = np.mean(neigh_slice)

    if return_details:
        details = {
            "outliers": outliers,
            "indices_outliers": outlier_indices,
            "local_score": local_score,
            "correction_applied": len(outlier_indices) > 0,
        }
        return y_data, details

    return y_data

def smoothing1d(ydata: pd.DataFrame, smoothing_fct: str = 'savgol', window_length: int = 5, sigma_or_polyorder: int = 2):
    """
    Apply a smoothing filter to the ydata to reduce noise. The default filter is the Savitzky-Golay filter.
    The window length and polynomial order (or sigma for Gaussian) can be adjusted.

    Possible Filters:
    - 'savgol': Savitzky-Golay filter, a polynomial smoothing filter using window length and polynomial order.
    - 'gaussian': Gaussian smoothing filter using window length and standard deviation sigma.
    - 'median': Median smoothing filter using window length.
    - 'None': No special smoothing filter applied. Only a rolling mean with the window length is used.

    Parameters
    ----------
    ydata : pandas.DataFrame
        Output value(s) of the dataset (typically named y in functions).
    smoothing_fct : str, optional
        Smoothing function to be applied. The default is 'savgol'.
    window_length : int, optional
        Length of the window in which the smoothing is applied. The default is 5.
    sigma_or_polyorder : int, optional
        Standard deviation of the Gaussian smoothing or the polynomial order of the Savitzky-Golay filter. 
        The default is 2 for Savitzky-Golay. For Gaussian, a value of 1 is recommended.

    Returns
    -------
    y_data : pandas.Series
        Smoothed ydata.

    Raises
    ------
    ValueError
        If the smoothing function is not recognized.

    Examples
    --------
    >>> smoothing1d(pd.DataFrame([1, 2, 3, 4, 5]), 'savgol', 5, 2)
    0    1.0
    1    2.0
    2    3.0
    3    4.0
    4    5.0
    dtype: float64
    """
    _check_array_like(ydata, 'ydata')
    try:
        ydata = pd.Series(ydata.squeeze())
    except Exception as e:  # noqa: BLE001
        raise TypeError(f'Error converting ydata to pandas Series: {e}')
    
    if str(smoothing_fct) == 'savgol':
        # Apply Savitzky-Golay filter
        smoothed = savgol_filter(ydata, window_length, polyorder=sigma_or_polyorder, mode='nearest')
        return pd.Series(smoothed, index=ydata.index)
    elif str(smoothing_fct) == 'gaussian':
        smoothed = ydata.rolling(window_length, win_type='gaussian', center=True).mean(std=sigma_or_polyorder)
        smoothed = smoothed.interpolate(method='nearest', limit_direction='both')
        return smoothed.ffill().bfill() #forward and backward filling of NaN values at the beginning and end of the series as fallback if interpolation fails
    elif str(smoothing_fct) == 'median':
        smoothed = ydata.rolling(window_length, center=True).median()
        smoothed = smoothed.interpolate(method='nearest', limit_direction='both')
        return smoothed.ffill().bfill() #forward and backward filling of NaN values at the beginning and end of the series as fallback if interpolation fails
    elif str(smoothing_fct) == 'None' or smoothing_fct is None:
        smoothed = ydata.rolling(window_length, center=True).mean()
        smoothed = smoothed.interpolate(method='nearest', limit_direction='both')
        return smoothed.ffill().bfill() #forward and backward filling of NaN values at the beginning and end of the series as fallback if interpolation fails
    else:
        raise ValueError(f'Smoothing function not recognized. Please choose between savgol, gaussian, median, or None, not {smoothing_fct}')
    

def rmv_opening(
    ydata: list | pd.DataFrame | pd.Series | np.ndarray, 
    sat_region: float = 0.05, 
    return_details: bool = False
):
    """
    Remove the opening in a hysteresis loop by assuming the same magnetization state at the beginning and end of the loop.

    This function calculates the mean of the first and last `sat_region` of points and compares their difference to their 
    noise levels. If the difference is larger than the noise level, a time/point number dependent slope is calculated 
    and subtracted from the ydata to correct the opening.

    In simpler terms: If the second branch of the hysteresis does not meet the first branch at the end (opening), 
    this function corrects it by assuming a linear slope over time or point number. This is useful if the light source 
    changes over time or if the sample field of view (FOV) moves.

    Parameters
    ----------
    ydata : list | pd.DataFrame | pd.Series | np.ndarray
        Output value(s) of the dataset (typically named y in functions).
    sat_region : float, optional
        Amount of points at the beginning and end of the hysteresis to be taken into account for the mean calculation.
        The default is 0.05, which represents the first and last 5% of points.
        This is equal to sat_regions=0.1 in the other functions for symmetric loops, where the saturation is calculated based on the applied field.
    return_details : bool, optional
        If True, returns additional details about the correction process. The default is False.

    Returns
    -------
    y_data : np.ndarray
        Array of magnetization M or intensity with an opening below the noise level at the end of the hysteresis. 
        Does not change the opening if no significant opening (relative to the noise) is present.

    Raises
    ------
    TypeError
        If `sat_region` is greater than or equal to 0.5.
        If `ydata` is not of a supported type.

    Examples
    --------
    >>> rmv_opening([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.1)
    array([1. , 1.9, 2.8, 3.7, 4.6, 5.5, 6.4, 7.3, 8.2, 9.1])
    """
    if sat_region >= 0.5:
        raise ValueError('Amount must be smaller than 0.5 (half of the hysteresis)')
    
    _check_array_like(ydata, 'ydata')
    
    sat_points = int(sat_region * len(ydata))
    
    # take difference of mean of first and last sat_region of ydata points
    mean_diff = np.mean(ydata[:sat_points]) - np.mean(ydata[-sat_points:])
    standard_deviation = np.std(ydata[:sat_points]) + np.std(ydata[-sat_points:])

    # check if difference is below the sum of both standard deviations (noise level) and therefore no
    # significant opening is present
    if np.abs(mean_diff) < standard_deviation:
        return ydata
 
    # calculate slope of opening with respect to the length of the hysteresis
    # ignore the outermost 2*0.5*sat_region of points, since the diff represents the mean at the outermost 0.5*sat_region positions
    opening_slope = - mean_diff / (len(ydata) - sat_points)

    # check if difference is below the sum of both standard deviations (noise level) and therefore no
    # significant opening is present
    if np.abs(mean_diff) < standard_deviation:
        if return_details:
            return ydata, {
                'opening_slope': opening_slope,
                'mean_diff': mean_diff,
                'standard_deviation': standard_deviation,
                'sat_points': sat_points,
                'correction_applied': False,
                }
        return ydata

    
    # subtract slope from ydata. Reminder slope is in time/point number and not in xdata/field strength
    y_data = ydata - opening_slope * np.arange(len(ydata))
    
    if return_details:
        return y_data, {
            'opening_slope': opening_slope, 
            'mean_diff': mean_diff,
            'standard_deviation': standard_deviation,
            'sat_points': sat_points,
            'correction_applied': True,
            }
    return y_data

def slope_correction(
    xdata: list | pd.DataFrame | pd.Series | np.ndarray,
    ydata: list | pd.DataFrame | pd.Series | np.ndarray,
    sat_region: float = 0.1,
    noise_threshold: float = 3,
    branch_difference: float = 0.3,
    return_details: bool = False
    ):
    """
    Corrects the slope in a hysteresis loop by assuming saturation in the outermost regions of the xdata.
    Fits a linear function to these parts to extract their averaged slope, which is then subtracted from the whole loop,
    if the slope is significant (above the noise level).

    Parameters
    ----------
    xdata : list | numpy.ndarray | pandas.DataFrame | pandas.Series
        Input value(s) of the dataset (typically named x in functions).
    ydata : list | numpy.ndarray | pandas.DataFrame | pandas.Series
        Output value(s) of the dataset (typically named y in functions).
    sat_region : float, optional
        Outermost fraction of the xdata assumed to be in saturation. Default is 0.1 (10%).
    noise_threshold : float, optional
        Threshold for the slope to be considered as noise. If the slope is below this threshold times the noise level,
        it is considered as noise and not subtracted. Default is 3.
    branch_difference : float, optional
        Maximum difference between the slopes of both branches. If the difference is larger, the function will not subtract the slope.
        The difference is based on the deviation from 1 as a ratio. Default is 0.3.
    return_details : bool, optional
        If True, returns additional details about the correction process. The default is False.

    Returns
    -------
    y_data : np.ndarray
        Array of magnetic moment M or intensity without a constant slope, i.e., flat at the border regions if no higher-order effects are present.

    Raises
    ------
    TypeError
        If `xdata` or `ydata` is not of a supported type.
    ValueError
        If no saturation region is found.

    Examples
    --------
    >>> slope_correction([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    array([1.9, 2.9, 3.9, 4.9, 5.9])
    """
    _check_array_like(xdata, 'xdata')
    _check_array_like(ydata, 'ydata')

    # convert to numpy arrays if not already
    x_data = np.asarray(xdata).copy()
    y_data = np.asarray(ydata).copy()

    # define sat_region in field strength
    sat_field_range = sat_region * (np.max(x_data) - np.min(x_data))
    # take end regions of hysteresis (saturated regions) by calculating the field
    # strengths which are assumed to be in saturation
    upper_saturation_limit = np.max(x_data) - sat_field_range
    lower_saturation_limit = np.min(x_data) + sat_field_range

    # take end regions of hysteresis (saturated regions of both branches)
    upper_saturation_region = x_data > upper_saturation_limit
    lower_saturation_region = x_data < lower_saturation_limit

    if not np.any(upper_saturation_region) or not np.any(lower_saturation_region):
        raise ValueError('No saturation region found')
    
    # fit linear functions to both ends
    popt1, _pcov1 = curve_fit(linear, x_data[upper_saturation_region], y_data[upper_saturation_region])
    popt2, _pcov2 = curve_fit(linear, x_data[lower_saturation_region], y_data[lower_saturation_region])
    # calculated mean slope of both hysteresis ends
    slope = np.mean([popt1[0], popt2[0]])

    # Calculate noise level (as standard deviation of the residuals)
    residuals_upper = y_data[upper_saturation_region] - linear(x_data[upper_saturation_region], *popt1)
    residuals_lower = y_data[lower_saturation_region] - linear(x_data[lower_saturation_region], *popt2)
    noise_level = noise_threshold * np.mean([np.std(residuals_upper), np.std(residuals_lower)])

    # Check if the slope (effect over the field range) is insignificant (below the noise level) and do nothing if it is
    slope_effect = np.abs(slope) * (np.max(x_data) - np.min(x_data))
    # norm_slope_diff is the normalized difference between the slopes of both branches. If both branches have the same slope, the ratio is 1 and the norm_slope_diff is 0. 
    # The larger their relative difference, the larger the norm_slope_diff. If the slope of the second branch is 0, the norm_slope_diff is set to infinity to avoid division by zero.
    norm_slope_diff = np.abs(1 - popt1[0]/popt2[0]) if popt2[0] != 0 else np.inf
    if slope_effect < noise_level or norm_slope_diff > branch_difference:
        if return_details:
            return y_data, {
                'slope': slope, 
                'noise_level': noise_level,
                'slope_effect': slope_effect,
                'norm_slope_diff': norm_slope_diff,
                'correction_applied': False
                }
        return y_data

    y_data_corrected = y_data - slope * x_data
    # otherwise return subtracted/corrected magnetization
    if return_details:
            return y_data_corrected, {
                'slope': slope, 
                'noise_level': noise_level,
                'slope_effect': slope_effect,
                'norm_slope_diff': norm_slope_diff,
                'correction_applied': True
                }

    return y_data_corrected

#TODO: Review this function
def quadratic_slope_correction(
    xdata: list | pd.DataFrame | pd.Series | np.ndarray,
    ydata: list | pd.DataFrame | pd.Series | np.ndarray,
    sat_region: float = 0.1,
    noise_threshold: float = 3,
    branch_difference: float = 0.3,
    return_details: bool = False
    ):
    """
    Corrects a smooth background in a hysteresis loop by iteratively minimizing the slope in the outer saturation regions.

    A quadratic background with coefficients ``a``, ``b`` and ``c`` is optimized so that the linear slope in the
    saturation windows becomes as small as possible. The optimization starts from a linear fit and runs for at most
    100 iterations. It stops early as soon as the remaining slope effect falls below the estimated noise level.

    Parameters
    ----------
    xdata : list | numpy.ndarray | pandas.DataFrame | pandas.Series
        Input value(s) of the dataset (typically named x in functions).
    ydata : list | numpy.ndarray | pandas.DataFrame | pandas.Series
        Output value(s) of the dataset (typically named y in functions).
    sat_region : float, optional
        Outermost fraction of the xdata assumed to be in saturation. Default is 0.1 (10%).
    noise_threshold : float, optional
        Threshold for the slope effect to be considered as noise. If the effect is below this threshold times the
        noise level, it is considered as noise and not subtracted. Default is 3.
    branch_difference : float, optional
        Maximum normalized difference between the slopes of both saturation regions. If the difference is larger, the
        function will not subtract the background. Default is 0.3.
    return_details : bool, optional
        If True, returns additional details about the correction process. The default is False.

    Returns
    -------
    y_data : np.ndarray
        Array of magnetic moment M or intensity without the optimized background.
        
    Raises
    ------
    TypeError
        If `xdata` or `ydata` is not of a supported type.
    ValueError
        If no saturation region is found or if there are not enough points in the saturation regions for a reliable fit.
    """
    _check_array_like(xdata, 'xdata')
    _check_array_like(ydata, 'ydata')

    x_data = np.asarray(xdata).copy()
    y_data = np.asarray(ydata).copy()

    sat_field_range = sat_region * (np.max(x_data) - np.min(x_data))
    upper_saturation_limit = np.max(x_data) - sat_field_range
    lower_saturation_limit = np.min(x_data) + sat_field_range

    upper_saturation_region = x_data > upper_saturation_limit
    lower_saturation_region = x_data < lower_saturation_limit

    if not np.any(upper_saturation_region) or not np.any(lower_saturation_region):
        raise ValueError('No saturation region found')

    if np.sum(upper_saturation_region) < 2 or np.sum(lower_saturation_region) < 2:
        raise ValueError('Not enough saturation points found for a minimization-based correction')

    # start from a linear fit to the saturation regions
    popt1, _pcov1 = curve_fit(linear, x_data[upper_saturation_region], y_data[upper_saturation_region])
    popt2, _pcov2 = curve_fit(linear, x_data[lower_saturation_region], y_data[lower_saturation_region])
    start_slope = np.mean([popt1[0], popt2[0]])
    start_offset = np.mean([popt1[1], popt2[1]])
    start_params = np.array([0.0, start_slope, start_offset], dtype=float)

    field_range = np.max(x_data) - np.min(x_data)

    def _evaluate_background(coefficients):
        background = quadratic(x_data, *coefficients)
        corrected = y_data - background

        upper_coeffs = np.polyfit(x_data[upper_saturation_region], corrected[upper_saturation_region], 1)
        lower_coeffs = np.polyfit(x_data[lower_saturation_region], corrected[lower_saturation_region], 1)

        residuals_upper = corrected[upper_saturation_region] - linear(x_data[upper_saturation_region], *upper_coeffs)
        residuals_lower = corrected[lower_saturation_region] - linear(x_data[lower_saturation_region], *lower_coeffs)

        slope_effect = np.mean([np.abs(upper_coeffs[0]), np.abs(lower_coeffs[0])]) * field_range
        noise_level = noise_threshold * np.mean([np.std(residuals_upper), np.std(residuals_lower)])

        if np.isclose(lower_coeffs[0], 0.0):
            norm_slope_diff = np.inf if not np.isclose(upper_coeffs[0], 0.0) else 0.0
        else:
            norm_slope_diff = np.abs(1 - upper_coeffs[0] / lower_coeffs[0])

        objective = slope_effect
        if norm_slope_diff > branch_difference:
            objective += (norm_slope_diff - branch_difference) * field_range

        return {
            'corrected': corrected,
            'slope_effect': slope_effect,
            'noise_level': noise_level,
            'norm_slope_diff': norm_slope_diff,
            'objective': objective,
            'upper_coeffs': upper_coeffs,
            'lower_coeffs': lower_coeffs,
        }

    state = {'best_metrics': None}

    def _objective(coefficients):
        metrics = _evaluate_background(coefficients)
        state['best_metrics'] = metrics
        return metrics['objective']

    def _callback(coefficients):
        metrics = _evaluate_background(coefficients)
        state['best_metrics'] = metrics
        if metrics['slope_effect'] <= metrics['noise_level']:
            raise StopIteration

    result = minimize(
        _objective,
        start_params,
        method='Nelder-Mead',
        callback=_callback,
        options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-12},
    )

    final_metrics = _evaluate_background(result.x)
    converged = final_metrics['slope_effect'] <= final_metrics['noise_level'] and final_metrics['norm_slope_diff'] <= branch_difference

    details = {
        'coefficients': result.x,
        'start_coefficients': start_params,
        'slope_effect': final_metrics['slope_effect'],
        'noise_level': final_metrics['noise_level'],
        'norm_slope_diff': final_metrics['norm_slope_diff'],
        'iterations': result.nit,
        'aborted_after_maxiter': result.nit >= 100 and not converged,
        'converged': converged,
        'correction_applied': converged,
        'optimizer_success': result.success,
        'optimizer_message': result.message,
    }

    if not converged:
        if return_details:
            return y_data, details
        return y_data

    if return_details:
        return final_metrics['corrected'], details

    return final_metrics['corrected']

def hys_norm(
    xdata: list | pd.DataFrame | pd.Series | np.ndarray,
    ydata: list | pd.DataFrame | pd.Series | np.ndarray,
    sat_region: float = 0.1,
    return_details: bool = False
    ):
    """
    Normalize a hysteresis loop by assuming saturation in the outermost regions of the xdata.

    This function takes a hysteresis loop with xdata (typically external magnetic field H) and ydata 
    (typically magnetic moment µ or intensity) as input. It assumes saturation in the outermost 
    `sat_region` (default 10%) of the xdata and calculates the average of these regions. This average 
    is then used to shift the ydata to its center (y_bias) and then divide it by the range in which 
    the hysteresis appears (norm). Recommended for hystereses with non-absolute values: e.g., MOKE/Kerr.

    Parameters
    ----------
    xdata : list | numpy.ndarray | pandas.DataFrame | pandas.Series
        List of externally applied field strengths H.
    ydata : list | numpy.ndarray | pandas.DataFrame | pandas.Series
        List of magnetization values M.
    sat_region : float, optional
        Outermost fraction of the xdata which is assumed to be in saturation.
        Default is 0.1, i.e., 10% of the outermost xdata is assumed to be in saturation.
    return_details : bool, optional
        If True, returns additional details about the normalization process. The default is False.

    Returns
    -------
    y_data : np.ndarray
        Normalized list of magnetization in the range of roughly -1 to +1.

    Raises
    ------
    TypeError
        If `xdata` or `ydata` is not of a supported type.
    ValueError
        If no saturation region is found.

    Examples
    --------
    >>> hys_norm([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    array([-1. , -0.5,  0. ,  0.5,  1. ])
    """
    _check_array_like(xdata, 'xdata')
    _check_array_like(ydata, 'ydata')

    # convert to numpy arrays if not already
    x_data = np.array(xdata).copy()
    y_data = np.array(ydata).copy()
    
    # take end regions of hysteresis (saturated regions)
    upper_saturation_limit = (1 - sat_region) * np.max(x_data)
    lower_saturation_limit = (1 - sat_region) * np.min(x_data)

    # take end regions of hysteresis (saturated regions)
    upper_saturation_region = x_data > upper_saturation_limit
    lower_saturation_region = x_data < lower_saturation_limit

    if not np.any(upper_saturation_region) or not np.any(lower_saturation_region):
        raise ValueError('No saturation region found')
    else:
        # average saturated regions
        lmax = np.mean(y_data[upper_saturation_region])
        lmin = np.mean(y_data[lower_saturation_region])
        # calculate shift/bias and normalization
        y_bias = 0.5 * (lmax + lmin)
        norm = 0.5 * (lmax - lmin)
        # return normalized magnetization
        y_data = (y_data - y_bias) / norm
        if return_details:
            return y_data, {'y_bias': y_bias, 'norm': norm}

        return y_data

def hys_center(
    xdata: list | pd.DataFrame | pd.Series | np.ndarray,
    ydata: list | pd.DataFrame | pd.Series | np.ndarray,
    sat_region: float = 0.1,
    normalize: bool = False,
    return_details: bool = False
    ):
    """
    Center and optionally normalize a hysteresis loop by assuming saturation in the outermost regions of the xdata.

    This function takes a hysteresis loop with xdata (typically external magnetic field H) and ydata 
    (typically magnetic moment µ or intensity) as input. The function assumes saturation in the outermost 
    `sat_region` (default 10%) of the xdata and calculates the average of these regions. This average 
    is then used to shift the ydata to its center (y_bias) and then divide it by the range 
    in which the hysteresis appears (norm) if `normalize` is True.
    Normalization is recommended for hystereses with non-absolute values (e.g., MOKE, not VSM).
    
    This function also compares the mean of the applied fields of both branches and shifts the branches to 
    the center if the difference is larger than one average step size. Which is recommended if the hysteresis 
    field was applied symmetrically but for example the data manipulation in the VSM software applied changes.

    Parameters
    ----------
    xdata : list | numpy.ndarray | pandas.DataFrame | pandas.Series
        List of externally applied field strengths H.
    ydata : list | numpy.ndarray | pandas.DataFrame | pandas.Series
        List of magnetization values M.
    sat_region : float, optional
        Outermost fraction of the xdata which is assumed to be in saturation.
        Default is 0.1, i.e., 10% of the outermost xdata is assumed to be in saturation.
    normalize : bool, optional
        If True, the ydata is normalized to the range of roughly -1 to +1.
        If False, the ydata is only centered around 0.
        The default is False.
    return_details : bool, optional
        If True, returns additional details about the centering and normalization process. The default is False.
        
    Returns
    -------
    tuple
        A tuple containing:
        - x_data : numpy.ndarray
            Adjusted list of externally applied field strengths H.
        - y_data : numpy.ndarray
            Centered (and optionally normalized) list of magnetization values M.

    Raises
    ------
    TypeError
        If `xdata` or `ydata` is not of a supported type.
    ValueError
        If `xdata` and `ydata` do not have the same length.
        If no saturation region is found.

    Examples
    --------
    >>> hys_center([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    (array([1, 2, 3, 4, 5]), array([-1. , -0.5,  0. ,  0.5,  1. ]))
    """
    _check_array_like(xdata, 'xdata')
    _check_array_like(ydata, 'ydata')

    # convert to numpy arrays if not already
    x_data = np.asarray(xdata).copy()
    y_data = np.asarray(ydata).copy()
    
    # x_data and y_data must have the same length and this length must be even
    if len(x_data) != len(y_data):
        raise ValueError('x_data and y_data must have the same length')
    # Check if the length of xdata is odd, i.e., the center point contributes to both branches.
    # Duplicate the center point in this case so that both branches are equally long.
    if len(x_data) % 2 != 0:
        center_index = len(x_data) // 2
        x_data = np.insert(x_data, center_index, x_data[center_index])
        y_data = np.insert(y_data, center_index, y_data[center_index])

    # Center in ydata: center between saturations

    # take end regions of hysteresis (saturated regions)
    upper_saturation_limit = (1 - sat_region) * np.max(x_data)
    lower_saturation_limit = (1 - sat_region) * np.min(x_data)
    upper_saturation_region = x_data > upper_saturation_limit
    lower_saturation_region = x_data < lower_saturation_limit

    if not np.any(upper_saturation_region) or not np.any(lower_saturation_region):
        raise ValueError('No saturation region found')
    else:
        # average saturated regions
        lmax = np.mean(y_data[upper_saturation_region])
        lmin = np.mean(y_data[lower_saturation_region])
        # calculate shift/bias and normalization
        y_bias = 0.5 * (lmax + lmin)
        norm = 0.5 * (lmax - lmin)
        if normalize:
            # return normalized magnetization
            y_data = (y_data - y_bias) / norm
        else:
            # return centered magnetization
            y_data = y_data - y_bias

    # Center in xdata: center branches if difference is above one step size
    
    # Calculate the step size of the xdata
    xdata_step = np.mean(np.abs(np.diff(x_data)))
    # Calculate the difference between the branches
    diff = np.mean(y_data[:len(y_data)//2]) - np.mean(y_data[len(y_data)//2:]) # if < 0, branch 1 is higher than branch 2

    x_shift = 0.0
    # Check if the difference between the branches is larger than one step size
    if np.abs(diff) > xdata_step:
        # Calculate the shift in xdata
        x_shift = diff / 2
        # Shift xdata
        x_data[:len(x_data)//2] -= x_shift
        x_data[len(x_data)//2:] += x_shift

    if return_details:
        return x_data, y_data, {'x_shift': x_shift, 'y_bias': y_bias, 'norm': norm}
    else:
        return x_data, y_data

# A one call function to prepare the hysteresis data for evaluation
def hys_prep(
    xdata: list | pd.DataFrame | pd.Series | np.ndarray,
    ydata: list | pd.DataFrame | pd.Series | np.ndarray,
    del_outlier_props: dict | None = None,
    rmv_opening_props: dict | None = None,
    slope_correction_props: dict | None = None,
    hys_norm_props: dict | None = None,
    return_details: bool = False
):
    """
    A one-call function to prepare hysteresis data for evaluation by applying a series of preprocessing steps in a meaningful order. The function applies outlier removal, opening removal, slope correction, and centering/normalization of the hysteresis loop. It is designed to streamline the preprocessing of hysteresis data, making it ready for further analysis. Each funcction can be controlled by passing a dictionary of parameters to the respective preprocessing step. In principle each function tests for the presence of the effect and only applies the correction if it is significant, whereby significance is defined by the respective props parameters.
    
    Parameters
    ----------
    xdata : list | numpy.ndarray | pandas.DataFrame | pandas.Series
        List of externally applied field strengths H.
    ydata : list | numpy.ndarray | pandas.DataFrame | pandas.Series
        List of magnetization values M.
    del_outlier_props : dict, optional
        Dictionary of parameters for the `del_outliers` function. Default is {'threshold': 5.0, 'neighbours': 10}.
    rmv_opening_props : dict, optional
        Dictionary of parameters for the `rmv_opening` function. Default is {'sat_region': 0.05}.
    slope_correction_props : dict, optional
        Dictionary of parameters for the `slope_correction` function. Default is {'sat_region': 0.1, 'noise_threshold': 3, 'branch_difference': 0.3}.
    hys_norm_props : dict, optional
        Dictionary of parameters for the `hys_norm` function. Default is {'sat_region': 0.1}.
        
    Returns
    -------
    tuple
        A tuple containing:
        - x_data : numpy.ndarray
            Adjusted list of externally applied field strengths H after preprocessing.
        - y_data : numpy.ndarray
            Preprocessed list of magnetization values M after outlier removal, opening removal, slope correction, and centering/normalization.
        - details : dict
            Dictionary containing details about each preprocessing step, including whether corrections were applied and relevant metrics.
            
    Raises
    ------
    TypeError
        If `xdata` or `ydata` is not of a supported type.
    ValueError
        If `xdata` and `ydata` do not have the same length.
    
    """
    # Check if xdata and ydata are the correct entry and have the same length
    _check_array_like(xdata, 'xdata')
    _check_array_like(ydata, 'ydata')
    
    if len(xdata) != len(ydata):
        raise ValueError('xdata and ydata must have the same length')

    # Avoid mutable defaults while keeping stable default preprocessing settings.
    if del_outlier_props is None:
        del_outlier_props = {'threshold': 5.0, 'neighbours': 10}
    if rmv_opening_props is None:
        rmv_opening_props = {'sat_region': 0.05}
    if slope_correction_props is None:
        slope_correction_props = {'sat_region': 0.1, 'noise_threshold': 3, 'branch_difference': 0.3}
    if hys_norm_props is None:
        hys_norm_props = {'sat_region': 0.1}
    
    if return_details:
        # Apply outlier removal
        y_data, outlier_details = del_outliers(ydata, **del_outlier_props, return_details=True)
        # Apply opening removal
        y_data, opening_details = rmv_opening(y_data, **rmv_opening_props, return_details=True)
        # Apply slope correction
        y_data, slope_details = slope_correction(xdata, y_data, **slope_correction_props, return_details=True)
        # Apply normalization
        y_data, centering_details = hys_norm(xdata, y_data, **hys_norm_props, return_details=True)
        
        details = {
            'outlier_removal': outlier_details,
            'opening_removal': opening_details,
            'slope_correction': slope_details,
            'centering_normalization': centering_details
        }
        return xdata, y_data, details
    else:
        # Apply outlier removal
        y_data = del_outliers(ydata, **del_outlier_props)
        # Apply opening removal
        y_data = rmv_opening(y_data, **rmv_opening_props)
        # Apply slope correction
        y_data = slope_correction(xdata, y_data, **slope_correction_props)
        # Apply normalization
        y_data = hys_norm(xdata, y_data, **hys_norm_props)
        
        return xdata, y_data
    

#%%
###############################################################################        
# 4. Data Evaluation
###############################################################################

def x_sect(xdata: pd.Series | np.ndarray, ydata: pd.Series | np.ndarray, offset: float = 0, steepness_for_fit: bool = False):
    """
    Calculate the first intersection of a hysteresis loop with the x-axis.

    This function takes a hysteresis loop with xdata (external field H) and ydata (magnetization M), 
    typically of a single branch, and calculates the first intersection with the x-axis using linear 
    interpolation between two subsequent points with changing sign of their y-values.

    Note: This function may be non-robust to strong noise. Consider using functions that check for a 
    clear change in sign by comparing the closest neighbours.

    Parameters
    ----------
    xdata : pd.Series | np.ndarray
        List of externally applied field strengths H, typically of a single branch.
    ydata : pd.Series | np.ndarray
        List of magnetization values M, typically of a single branch.
    offset : float, optional
        Offset for the x-axis intersection. The default is 0.
    steepness_for_fit : bool, optional
        If True, the function also returns the steepness of the linear fit. The default is False.

    Returns
    -------
    intersect : float
        x-value of the intersection with the x-axis (HC). Returns 0 if no intersection is found.
    intersect_err : float
        Uncertainty of the x-value of the intersection with the x-axis (dHC). Returns 0 if no intersection is found.
    a : float, optional
        Steepness of the linear fit. Only returned if `steepness_for_fit` is True.

    Raises
    ------
    TypeError
        If `xdata` or `ydata` is not of a supported type.
    ValueError
        If `xdata` and `ydata` do not have the same length.

    Examples
    --------
    >>> x_sect(pd.Series([1, 2, 3]), pd.Series([-1, 0, 1]))
    (2.0, 0.0)
    >>> x_sect(pd.Series([1, 2, 3]), pd.Series([-1, 0, 1]), offset=0.5, steepness_for_fit=True)
    (2.0, 0.0, 1.0)
    """
    
    if len(xdata) != len(ydata):
        raise ValueError('xdata and ydata must have the same length')
    
    _check_array_like(xdata, 'xdata')
    _check_array_like(ydata, 'ydata')

    if np.abs(offset) > np.max(np.abs(ydata)):
        raise ValueError('Offset is larger than the maximum absolute value of ydata, no intersection with the offset can be found')
    
    x_data = np.asarray(xdata).copy()
    y_data = np.asarray(ydata).copy() - offset # shift ydata by offset to find intersection with the offset instead of the x-axis
    
    # Ensure that the ydata list starts with a negative value (negative saturation)
    if next(iter(y_data)) > 0.0: 
        x_data = np.flipud(x_data)
        y_data = np.flipud(y_data)

    # Initialize variables
    intersect = 0.0
    intersect_err = 0.0
    a = 0.0

    # Check for points where the product of two adjacent points is negative or equal to zero
    for i in range(1, len(y_data)):
        product = y_data[i-1] * y_data[i]
        if product <= 0:
            # If the product is zero, the intersection is directly found
            if y_data[i] == 0:
                return x_data[i], 0.0
            else:
                # Linearly interpolate between the two points
                a = (y_data[i] - y_data[i-1]) / (x_data[i] - x_data[i-1])
                if a == np.inf or a == -np.inf: # Rarely, x_data[i] and x_data[i-1] are identical leading to a division by zero. Then the slope is wrongly calculated as inf or -inf. Happend once in 2 years of usage.
                    a = (y_data[i] - y_data[i-2]) / (x_data[i] - x_data[i-2])
                if a != 0:
                    b = y_data[i-1] - a * x_data[i-1]
                    intersect = -b / a
                    intersect_err = max(np.abs(intersect - x_data[i]), np.abs(intersect - x_data[i-1]))

    if steepness_for_fit:
        return intersect, intersect_err, a
    else:
        return intersect, intersect_err
    
def y_sect(xdata: pd.Series | np.ndarray, ydata: pd.Series | np.ndarray, offset: float = 0):
    """
    Calculate the first intersection of a hysteresis loop with the y-axis.

    This function takes a hysteresis loop with xdata (external field H) and ydata 
    (magnetization M), typically of a single branch, and calculates the first 
    intersection with the y-axis using linear interpolation between two subsequent 
    points with changing sign of their x-values.

    If there is an exchange bias field (HEB) present, it will shift all x-values by
    this field strength before performing the calculation. Thus the intersection
    with the HEB field strength is returned.

    Note: This function may be non-robust to strong noise. Consider using functions 
    that check for a clear change in sign by comparing the closest neighbours.

    Parameters
    ----------
    xdata : pd.Series | np.ndarray
        List of externally applied field strengths H, typically of a single branch.
    ydata : pd.Series | np.ndarray
        List of magnetization values M, typically of a single branch.
    offset : float, optional
        Offset for the y-axis intersection (typically HEB). The default is 0.

    Returns
    -------
    intersect : float
        y-value of the intersection with the y-axis (MR). Returns 0 if no intersection is found.
    intersect_err : float
        Uncertainty of the y-value of the intersection with the y-axis (dMR). Returns 0 if no intersection is found.

    Raises
    ------
    TypeError
        If `xdata` or `ydata` is not of a supported type.
    ValueError
        If `xdata` and `ydata` do not have the same length.
        If `offset` is larger than the maximum absolute value of `xdata`, no intersection with the offset can be found.

    Examples
    --------
    >>> y_sect(pd.Series([1, 2, 3]), pd.Series([-1, 0, 1]))
    (0.0, 0.0)
    >>> y_sect(pd.Series([1, 2, 3]), pd.Series([-1, 0, 1]), offset=1)
    (0.0, 0.0)
    """
    
    if len(xdata) != len(ydata):
        raise ValueError('xdata and ydata must have the same length')
    
    _check_array_like(xdata, 'xdata')
    _check_array_like(ydata, 'ydata')
    
    if np.abs(offset) > np.max(np.abs(xdata)):
        raise ValueError('Offset is larger than the maximum absolute value of xdata, no intersection with the offset can be found')
    
    x_data = np.asarray(xdata).copy() - offset # shift xdata by offset to find intersection with the offset instead of the y-axis
    y_data = np.asarray(ydata).copy()
    
    # Ensure that the xdata list starts with a negative value (from left to right)
    if next(iter(x_data)) > 0.0:
        y_data = np.flipud(y_data)
        x_data = np.flipud(x_data)

    # Initialize variables
    intersect = 0.0
    intersect_err = 0.0

    #  Check for points where the product of two adjacent points in x is negative or equal to zero
    for i in range(1, len(x_data)):
        product = x_data[i-1] * x_data[i]
        if product <= 0:
            # If the product is zero, the intersection is directly found
            if x_data[i] == 0:
                return y_data[i], 0.0
            else:
                # Linearly interpolate between the two points
                a = (y_data[i] - y_data[i-1]) / (x_data[i] - x_data[i-1])
                if a == np.inf or a == -np.inf: # Rarely, x_data[i] and x_data[i-1] are identical leading to a division by zero. Then the slope is wrongly calculated as inf or -inf. Happend once in 2 years of usage.
                    a = (y_data[i] - y_data[i-2]) / (x_data[i] - x_data[i-2])
                if a != 0:
                    b = y_data[i-1] - a * x_data[i-1]
                    intersect = b
                    intersect_err = max(np.abs(intersect - y_data[i]), np.abs(intersect - y_data[i-1]))
                    
    return intersect, intersect_err

def num_derivative(xdata: pd.Series | np.ndarray, ydata: pd.Series | np.ndarray):
    """
    Take an input dataset (xdata, ydata) and numerically calculate the 
    derivative of it by linear interpolation of the gradient in ydata between
    two xdata points. Works best with a dense dataset with a low amount of noise.
    Consider smoothing the data before using this function.

    Parameters
    ----------
    xdata : pd.Series or np.ndarray
        Input value(s) of the dataset (typically named x in functions).
    ydata : pd.Series or np.ndarray
        Output value(s) of the dataset (typically named y in functions).

    Returns
    -------
    der_xdata : np.ndarray
        Interpolated value(s) of dataset xdata, which are in between the input 
        data. E.g. xdata = [3, 5, 7], der_xdata = [4, 6]. Length is one less
        compared to xdata.
    der_ydata : np.ndarray
        Linearly interpolated slopes/derivatives of dataset ydata, which are 
        in between the input y- and x-data. 
        E.g. slope_i = (ydata_(i+1) - ydata_i) / (xdata_(i+1) - xdata_i). 

    Raises
    ------
    TypeError
        If `xdata` or `ydata` is not of a supported type.
    ValueError
        If `xdata` and `ydata` do not have the same length.

    Examples
    --------
    >>> num_derivative(pd.Series([3, 5, 7]), pd.Series([1, 2, 3]))
    (array([4., 6.]), array([0.5, 0.5]))
    """
    
    if len(xdata) != len(ydata):
        raise ValueError('xdata and ydata must have the same length')
    
    _check_array_like(xdata, 'xdata')
    _check_array_like(ydata, 'ydata')

    x_data = np.asarray(xdata).copy()
    y_data = np.asarray(ydata).copy()
    
    # Calculate the derivative
    der_ydata = np.diff(y_data) / np.diff(x_data)
    # Calculate the xdata in between the input xdata
    der_xdata = x_data[:-1] + np.diff(x_data) / 2
    
    return der_xdata, der_ydata

def num_integral(xdata: pd.Series | np.ndarray, ydata: pd.Series | np.ndarray):
    """
    Take an input dataset (xdata, ydata) and numerically calculate the 
    integral of it by linear interpolation of the area between two ydata points 
    and the xdata. Works best with a dense dataset.

    Note: This function has not yet been tested thoroughly. Please report any issues.
    The integral function in the evaluation functions are more feasible for hysteresis loops.

    Parameters
    ----------
    xdata : pd.Series or np.ndarray
        Input value(s) of the dataset (typically named x in functions).
    ydata : pd.Series or np.ndarray
        Output value(s) of the dataset (typically named y in functions).

    Returns
    -------
    int_xdata : np.ndarray
        Interpolated value(s) of dataset xdata, which are in between the input 
        data. E.g. xdata = [3, 5, 7], int_xdata = [4, 6]. Length is one less
        compared to xdata.
    int_ydata : np.ndarray
        Linearly interpolated areas/integrals of dataset ydata, which are 
        in between the input y- and x-data. 
        E.g. area_i = (ydata_(i+1) + ydata_i) / 2 * (xdata_(i+1) - xdata_i). 

    Raises
    ------
    TypeError
        If `xdata` or `ydata` is not of a supported type.
    ValueError
        If `xdata` and `ydata` do not have the same length.

    Examples
    --------
    >>> num_integral(pd.Series([3, 5, 7]), pd.Series([1, 2, 3]))
    (array([4., 6.]), array([2., 2.]))
    """
    if len(xdata) != len(ydata):
        raise ValueError('xdata and ydata must have the same length')
    
    _check_array_like(xdata, 'xdata')
    _check_array_like(ydata, 'ydata')

    x_data = np.asarray(xdata).copy()
    y_data = np.asarray(ydata).copy()
    
    # Calculate the integral
    int_ydata = (y_data[:-1] + y_data[1:]) / 2 * np.abs(np.diff(x_data))
    # Calculate the xdata in between the input xdata
    int_xdata = x_data[:-1] + np.diff(x_data) / 2
    
    return int_xdata, int_ydata

def lin_hyseval(
        xdata: list | pd.DataFrame | pd.Series | np.ndarray, 
        ydata: list | pd.DataFrame | pd.Series | np.ndarray,
        sat_region: float = SAT_COND,
        use_offset: bool = True,
        steepness_for_fit: bool = False):
    """
    Calculates the exchange bias field (HEB) and coercive field (HC) as well as their 
    uncertainties based on the x-axis intersection function `x_sect()`. The exchange bias 
    field and coercive field are calculated corresponding to their geometric relation to 
    these two intersections. Not suitable for double hysteresis or a hysteresis with a 
    large offset in magnetization. For such cases, centering or normalization is recommended.

    With an offset, the hysteresis loop can be shifted in the y-direction. This may enable 
    the calculation of the coercive field strength and the exchange bias field for several 
    loops (e.g., for a double hysteresis) by shifting the loops up or down. This method 
    requires either a lot of manual work or an algorithm to determine the offset. The offset 
    can be determined by shifting the hysteresis loop until a value is reached where the 
    magnetization at HEB (shifted) has an equal magnitude for both branches if both hysteresis 
    loops are completely separated. However, the double arctan method is able to calculate 
    more information and works already.

    Parameters
    ----------
    xdata : list | pd.DataFrame | pd.Series | np.ndarray
        List of externally applied field strengths H, typically of a single branch.
    ydata : list | pd.DataFrame | pd.Series | np.ndarray
        List of magnetization values M, typically of a single branch.
    offset : float, optional
        Offset to shift the hysteresis loop in the y-direction. Default is 0.0.
    steepness_for_fit : bool, optional
        If True, the function also returns the steepness of the linear fit. Default is False.

    Returns
    -------
    dict
        A dictionary containing:
        - HEB : float
            Exchange bias field strength.
        - dHEB : float
            Uncertainty of exchange bias field.
        - HC : float
            Coercive field strength, always positive.
        - dHC : float
            Uncertainty of coercive field strength.
        - MR : tuple
            Remanence at the exchange bias field, and the individual remanence values for both branches.
        - dMR : float
            Uncertainty of remanence.
        - MHEB : float
            Magnetization at the exchange bias field.
        - dMHEB : float
            Uncertainty of magnetization at the exchange bias field.
        - a1 : float, optional
            Steepness of the linear fit for the first branch. Only returned if `steepness_for_fit` is True.
        - a2 : float, optional
            Steepness of the linear fit for the second branch. Only returned if `steepness_for_fit` is True.

    Raises
    ------
    TypeError
        If `xdata` or `ydata` is not of a supported type.
    ValueError
        If `xdata` and `ydata` do not have the same length.

    Examples
    --------
    >>> lin_hyseval([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    {'HEB': 2.5, 'dHEB': 0.1, 'HC': 1.0, 'dHC': 0.1, 'MR': (0.5, 1.0, 0.0), 'dMR': 0.1, 'MHEB': 0.5, 'dMHEB': 0.1}
    >>> lin_hyseval([1, 2, 3, 4, 5], [2, 3, 4, 5, 6], steepness_for_fit=True)
    {'HEB': 2.5, 'dHEB': 0.1, 'HC': 1.0, 'dHC': 0.1, 'MR': (0.5, 1.0, 0.0), 'dMR': 0.1, 'MHEB': 0.5, 'dMHEB': 0.1, 'a1': 1.0, 'a2': 1.0}
    """
    
    _check_array_like(xdata, 'xdata')
    _check_array_like(ydata, 'ydata')

    x_data = np.asarray(xdata).copy()
    y_data = np.asarray(ydata).copy()

    # Check if the length of xdata is odd, i.e., the center point contributes to both branches.
    # Duplicate the center point in this case so that both branches are equally long.
    if len(x_data) % 2 != 0:
        center_index = len(x_data) // 2
        x_data = np.insert(x_data, center_index, x_data[center_index])
        y_data = np.insert(y_data, center_index, y_data[center_index])
    
    if use_offset:
        # take end regions of hysteresis (saturated regions)
        upper_saturation_limit = (1 - sat_region) * np.max(x_data)
        lower_saturation_limit = (1 - sat_region) * np.min(x_data)
        upper_saturation_region = x_data > upper_saturation_limit
        lower_saturation_region = x_data < lower_saturation_limit

        if not np.any(upper_saturation_region) or not np.any(lower_saturation_region):
            raise ValueError('No saturation region found')
        
        else:
            # average saturated regions
            magmax = np.mean(y_data[upper_saturation_region])
            magmin = np.mean(y_data[lower_saturation_region])
            # calculate shift/bias and normalization
            magoffset = float(0.5 * (magmax + magmin))
            _norm = 0.5 * (magmax - magmin)

    else:
        magoffset = 0.0

    # Obtain intersections as coercive fields with the x_sect function
    # Split the array into two halves using slicing
    mid_index = len(x_data) // 2
    Xdata1 = x_data[:mid_index]
    Xdata2 = x_data[mid_index:]

    Ydata1 = y_data[:mid_index]
    Ydata2 = y_data[mid_index:]
    # branch-dependently
    HC1, dHC1, a1 = x_sect(Xdata1, Ydata1, offset=magoffset, steepness_for_fit=True) # first branch
    HC2, dHC2, a2 = x_sect(Xdata2, Ydata2, offset=magoffset, steepness_for_fit=True) # second branch

    half_step_size = np.mean(np.abs(np.diff(x_data))) / 2
    
    # EB field as average of coercive fields/intersects
    HEB = (HC1 + HC2) / 2
    dHEB = (dHC1 + dHC2) / 2 + half_step_size # uncertainty via propagation of uncertainty
    # Coercive field as half of the distance between the two intersections
    HC = np.abs((HC1 - HC2) / 2)
    dHC = (dHC1 + dHC2) / 2 + half_step_size # uncertainty via propagation of uncertainty
    
    # Remanence at zero field strength
    MR1, dMR1 = y_sect(Xdata1, Ydata1, 0)
    MR2, dMR2 = y_sect(Xdata2, Ydata2, 0)
    # Average of both branches
    MR = ((np.abs(MR1) + np.abs(MR2)) / 2, MR1, MR2)
    dMR = (dMR1 + dMR2) / 2

    # Magnetization at the exchange bias field
    MHEB1, dMHEB1 = y_sect(Xdata1, Ydata1, offset=HEB)
    MHEB2, dMHEB2 = y_sect(Xdata2, Ydata2, offset=HEB)
    # Average of both branches
    MHEB = ((np.abs(MHEB1) + np.abs(MHEB2)) / 2, MHEB1, MHEB2)
    dMHEB = (dMHEB1 + dMHEB2) / 2

    params = {
        'HEB': HEB,
        'dHEB': dHEB,
        'HC': HC,
        'dHC': dHC,
        'MR': MR,
        'dMR': dMR,
        'MHEB': MHEB,
        'dMHEB': dMHEB,
    }

    if steepness_for_fit:
        params['a1'] = a1
        params['a2'] = a2
    
    return params

def arctan_hyseval(
    xdata: list | pd.DataFrame | pd.Series | np.ndarray, 
    ydata: list | pd.DataFrame | pd.Series | np.ndarray,
    sat_cond: float = SAT_COND,
    sat_region: float = SAT_COND,
    use_offset: bool = True,
    param_estimates: dict | None = None,
    param_bounds: dict | None = None,
    param_fixed: dict | None = None,
    method: str = 'leastsq',):
    """
    Fits a hysteresis loop with an arctan function to extract several parameters
    including the exchange bias field, coercive field strength, remanence, saturation
    magnetization, saturation field strength, the slopes at the intersection with the
    x-axis (HC field) and the EB field, a possible offset, the enclosed area and 
    enclosed angle of the hysteresis loop, and the rectangularity of the hysteresis.
    For all parameters, the uncertainty is calculated via propagation of uncertainty.

    It also provides the fitted ydata and its uncertainty as well as the lmfit result.

    Note: The function is not suitable for double hysteresis loops.
    
    Parameters
    ----------
    xdata : list | pd.DataFrame | pd.Series | np.ndarray
        List of externally applied field strengths H.
    ydata : list | pd.DataFrame | pd.Series | np.ndarray
        List of magnetization values M.
    sat_cond : float, optional
        Saturation condition for the tanh function. Default is SAT_COND.
    param_estimates : dict, optional
        Initial guesses for the fit parameters. Default is None.
        'a': global offset, 'b': amplitude (2 * MS / pi), 'c': steepness, 'd': global shift (HEB), 'e': branch offset (HC)
    param_bounds : dict, optional
        Bounds for the fit parameters, has to be a dictionary of lists of length 2.
        The first element is the lower bound, the second element is the upper bound.
        Default is None.
    param_fixed : dict, optional
        Whether a parameter is fixed to its estimate (True) or not (False). 
        If param_estimates is not provided, the param is fixed to a numerical calculated estimate.
        Default is None.

    Returns
    -------
    tuple
        A tuple containing:
        - fitted_data : dict
            Dictionary containing:
            - xdata : pd.Series
                Interpolated xdata values.
            - ydata : pd.Series
                Fitted ydata values.
            - xdata_err : pd.Series
                Uncertainty of xdata values.
            - ydata_err : pd.Series
                Uncertainty of ydata values.
        - params : dict
            Dictionary containing:
            - r_squared : float
                Coefficient of determination of the fit.
            - HEB : float
                Exchange bias field strength.
            - dHEB : float
                Uncertainty of exchange bias field.
            - HC : float
                Coercive field strength.
            - dHC : float
                Uncertainty of coercive field strength.
            - MS : float
                Saturation magnetization.
            - dMS : float
                Uncertainty of saturation magnetization.
            - MS : float
                Saturation magnetization.
            - dMS : float
                Uncertainty of saturation magnetization.
            - MR : float
                Remanence at zero field strength.
            - dMR : float
                Uncertainty of remanence.
            - MHEB : float
                Magnetization at the exchange bias field.
            - dMHEB : float
                Uncertainty of magnetization at the exchange bias field.
            - integral : float
                Area of the hysteresis loop.
            - dintegral : float
                Uncertainty of the area of the hysteresis loop.
            - saturation_fields : tuple
                Saturation field strengths.
            - dsaturation_fields : tuple
                Uncertainty of saturation field strengths.
            - slope_atHC : float
                Slope at the coercive field.
            - dslope_atHC : float
                Uncertainty of the slope at the coercive field.
            - slope_atHEB : float
                Slope at the exchange bias field.
            - dslope_atHEB : float
                Uncertainty of the slope at the exchange bias field.
            - alpha : float
                Angle enclosed between the slopes at HC and HEB.
            - dalpha : float
                Uncertainty of the angle enclosed between the slopes at HC and HEB.
            - rectangularity : float
                Rectangularity of the hysteresis loop.
            - drectangularity : float
                Uncertainty of the rectangularity of the hysteresis loop.

    Raises
    ------
    TypeError
        If `xdata` or `ydata` is not of a supported type.
    ValueError
        If `xdata` and `ydata` do not have the same length.

    Examples
    --------
    >>> tan_hyseval([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    ({'xdata': ..., 'ydata': ..., 'xdata_err': ..., 'ydata_err': ...},
     {'r_squared': ..., 'HEB': ..., 'dHEB': ..., 'HC': ..., 'dHC': ..., 'MR': ..., 'dMR': ..., 'MHEB': ..., 'dMHEB': ..., 'integral': ..., 'dintegral': ..., 'saturation_fields': ..., 'dsaturation_fields': ..., 'slope_atHC': ..., 'dslope_atHC': ..., 'slope_atHEB': ..., 'dslope_atHEB': ..., 'alpha': ..., 'dalpha': ..., 'rectangularity': ..., 'drectangularity': ...})
    """
    _check_array_like(xdata, 'xdata')
    _check_array_like(ydata, 'ydata')

    x_data = np.asarray(xdata).copy()
    y_data = np.asarray(ydata).copy()
    
    # Check if the length of xdata is odd, i.e.the center point contributes to both branches.
    # Duplicate the center point in this case so that both branches are equally long.
    if len(x_data) % 2 != 0:
        center_index = len(x_data) // 2
        x_data = np.insert(x_data, center_index, x_data[center_index])
        y_data = np.insert(y_data, center_index, y_data[center_index])
    
    # Check if xdata and ydata have the same length
    if len(x_data) != len(y_data):
        raise ValueError('xdata and ydata must have the same length')
    
    # quick linear calculation to determine initial guesses for the exchange bias and the coercive field strength
    LIN = lin_hyseval(x_data, y_data, sat_region=sat_region, use_offset=use_offset, steepness_for_fit=True)
    HEB_tmp, HC_tmp = LIN['HEB'], LIN['HC']
    slope = (LIN['a1'] + LIN['a2']) # No average because the fit works better with a steeper initial slope
    
    # Create a model from the function
    model = Model(arctan_hys)

    # Quick explanation of the steepness parameter
    """
    arctan (x) = a + b * np.arctan(c * (x - d + e))
    a: offset
    b: Amplitude
    c: steepness
    d: global shift
    e: local shift (for each branch)
    slope: steepness of the arctan function at the intersection with the x-axis (x = d - e)

    f'(x) = (b * c) / (1 + (c * (x - d + e))**2)
    steepness at x = (-e +d) (so at the intersection with the x-axis)
    f'(x) = (b * c) / (1 + (c * (0))**2) = b * c = slope
    c = slope / b
    steepness at x = d
    f'(x) = (b * c) / (1 + (c * (d - d + e))**2) = (b * c) / (1 + (c * e)**2) = slope
    """

    # Define the parameters
    params = Parameters()
    params.add('a', value=(np.max(y_data) + np.min(y_data))/2) # offset
    params.add('b_1', value=(np.max(y_data) - np.min(y_data))/2) # amplitude
    params.add('c_1', value=slope / params['b_1'].value, min=0) # steepness
    params.add('d_1', value=HEB_tmp, min=np.min(x_data), max=np.max(x_data)) # exchange bias field
    params.add('e_1', value=HC_tmp, min=-(np.max(x_data) - np.min(x_data)), max=(np.max(x_data) - np.min(x_data))) # coercive field

    # Vary the parameters depending on additional input (normally not necessary)
    if param_estimates:
        for key, value in param_estimates.items():
            params[key].value = value
    if param_bounds:
        for key, value in param_bounds.items():
            params[key].min, params[key].max = value
    if param_fixed:
        for key, value in param_fixed.items():
            params[key].vary = not value 

    # Fit the model to the data
    result = model.fit(y_data, params, calc_covar=True, method=method, xdata=x_data,)

    # Calculate fitted magnetization values
    y_data_fitted = result.best_fit
    #convert to pd.Series for consistency
    y_data_fitted = pd.Series(y_data_fitted)

    # calculate the 3 sigma uncertainty of the fit
    y_data_fitted_err = result.eval_uncertainty(sigma=3)
    # convert to pd.Series for consistency
    y_data_fitted_err = pd.Series(y_data_fitted_err)

    # create a copy of the averaged xdata (branch 1 and 2) with which the results are
    # displayed. Only important for plotting the area of the hysteresis loop. As long as
    # the xdata is symmetric, the average is symmetric as well.
    # Split the array into two halves using slicing
    mid_index = len(x_data) // 2
    Xdata1 = x_data[:mid_index]
    Xdata2 = x_data[mid_index:]
    x_data_fitted = np.mean([Xdata2[::-1], Xdata1], axis=0)
    x_data_fitted = pd.Series([*x_data_fitted, *x_data_fitted[::-1]])

    #xdata uncertainty = half step size
    #TODO: For Kerr and VSM the step size may change, so this should be calculated from the data
    #TODO: np.diff without mean? In that case: HC/HEB error should be calculated at their respective positions
    #TODO: And I have to make sure that np.diff has the same length as xdata
    half_step_size = np.mean(np.abs(np.diff(x_data_fitted))) / 2
    #convert to pd.Series for consistency
    x_data_fitted_err = pd.Series([half_step_size] * len(x_data_fitted))

    fitted_data = {
        'xdata': x_data_fitted, 
        'ydata': y_data_fitted,
        'xdata_err': x_data_fitted_err,
        'ydata_err': y_data_fitted_err,
    }
    
    params = arctan_hyseval_params(result, x_data, y_data, sat_cond)
    
    return fitted_data, params, result

def _estimate_multi_hys_params(x_data: np.ndarray, y_data: np.ndarray, n: int, sat_region: float = SAT_COND, use_offset: bool = True):
    """
    Estimate n starting guesses each for HEB (d), HC (e), amplitude (b) and
    steepness (c), for a hysteresis loop made up of n overlapping arctan
    sub-loops (as fitted by `double_arctan_hyseval`/`mult_arctan_hyseval`).

    Instead of a single global linear-fit guess repeated with small, fixed
    offsets for every sub-loop (which the multi-arctan fits are sensitive to
    getting wrong, especially for HEB/HC), this locates each sub-loop
    directly from the data:
        - HEB (d) is estimated from the field position of each peak in the
          branch-separation curve M_decreasing(H) - M_increasing(H), i.e.
          where the two branches differ the most - each distinct sub-loop
          shows up as its own peak there.
        - HC (e) is estimated from the width of that same peak (via
          `scipy.signal.peak_widths`, which locates it from the local rate
          of change/gradient around the peak), since a sub-loop's field
          width in the branch-separation curve is ~2 * HC.
    This is a single pass of peak-finding over the existing data (no extra
    fitting), so it stays fast even for a strict n_arctan.

    If fewer distinct peaks are found than sub-loops requested (e.g. for a
    shallow or noisy loop), the remaining sub-loops fall back to evenly
    offset guesses around the single linear estimate, matching the previous
    behavior, so this never estimates fewer sub-loops than asked for.

    Parameters
    ----------
    x_data, y_data : numpy.ndarray
        Field and magnetization data, of even length (as ensured by the
        calling function), split at the halfway point into two branches.
    n : int
        Number of sub-loops (arctan terms) to estimate.
    sat_region, use_offset : optional
        Only used for the fallback estimate (see below); passed through to
        `lin_hyseval` unchanged from the calling function's own parameters.

    Returns
    -------
    d_est, e_est, b_est, c_est : list of float
        Length-n initial guesses for HEB, HC, amplitude, and steepness,
        sorted by increasing HEB.
    """
    mid = len(x_data) // 2
    H = np.mean([x_data[mid:][::-1], x_data[:mid]], axis=0)
    delta = y_data[mid:][::-1] - y_data[:mid]  # branch separation ("loop opening") vs. field

    # light smoothing so single-point noise doesn't masquerade as its own sub-loop
    window = max(1, (len(delta) // 50) | 1)  # odd window, ~2% of the data length
    delta_smooth = np.convolve(delta, np.ones(window) / window, mode='same') if window > 1 else delta

    peaks, properties = find_peaks(np.abs(delta_smooth), prominence=max(np.ptp(delta_smooth) * 0.05, 1e-9))
    step = np.mean(np.abs(np.diff(H)))

    if len(peaks) > 0:
        strongest = np.argsort(properties['prominences'])[::-1][:n]
        peaks = peaks[strongest]
        widths = peak_widths(np.abs(delta_smooth), peaks, rel_height=0.5)[0]
        d_found = H[peaks]
        e_found = widths * step / 2
        heights = np.abs(delta_smooth[peaks])
        order = np.argsort(d_found)  # sort by field position, matching the d_1 < d_2 < ... convention used elsewhere
        d_found, e_found, heights = d_found[order], e_found[order], heights[order]
    else:
        d_found, e_found, heights = np.array([]), np.array([]), np.array([])

    # fall back to the previous single linear-fit guess for any remaining (undetected) sub-loops
    n_found = len(d_found)
    if n_found < n:
        LIN = lin_hyseval(x_data, y_data, sat_region=sat_region, use_offset=use_offset)
        HEB_tmp = LIN['HEB'] if LIN['HEB'] is not None else np.mean(x_data)
        HC_tmp = LIN['HC'] if LIN['HC'] else (np.max(x_data) - np.min(x_data)) * 0.1
        n_missing = n - n_found
        offsets = (np.arange(n_missing) - (n_missing - 1) / 2) * 0.2 * (np.max(x_data) - np.min(x_data))
        fill_height = np.mean(heights) if n_found else np.ptp(y_data) / (2 * n)
        d_found = np.concatenate([d_found, HEB_tmp + offsets])
        e_found = np.concatenate([e_found, np.full(n_missing, HC_tmp)])
        heights = np.concatenate([heights, np.full(n_missing, fill_height)])
        order = np.argsort(d_found)
        d_found, e_found, heights = d_found[order], e_found[order], heights[order]

    # split the total amplitude between sub-loops proportionally to their peak height,
    # instead of assuming they contribute equally
    total_amplitude = (np.max(y_data) - np.min(y_data)) / 2
    total_height = np.sum(heights) or 1.0
    b_est = total_amplitude * heights / total_height

    # characteristic slope ~ peak height / peak width; c = slope / b (same relation as in arctan_hyseval)
    slopes = heights / np.maximum(e_found, step / 2)
    c_est = np.clip(slopes / np.maximum(b_est, 1e-6), 0.5, 100)

    return list(d_found), list(e_found), list(b_est), list(c_est)

def double_arctan_hyseval(
    xdata: list | pd.DataFrame | pd.Series | np.ndarray, 
    ydata: list | pd.DataFrame | pd.Series | np.ndarray, 
    sat_cond: float = SAT_COND,
    sat_region: float = SAT_COND,
    use_offset: bool = True,
    param_estimates: dict | None = None,
    param_bounds: dict | None = None,
    param_fixed: dict | None = None,
    method: str = 'leastsq',):
    """
    Fits a hysteresis loop with a double tanh function to extract several parameters
    including the exchange bias field, coercive field strength, remanence, saturation
    magnetization, saturation field strength, the slopes at the intersection with the
    x-axis and the EB field, the steepness of the tanh function, a possible offset, the
    enclosed area and angle of the hysteresis loop, and the rectangularity of the hysteresis.

    It also provides the fitted ydata and its uncertainty as well as the lmfit result.

    Note: For the uncertainties, the hysteresis loops are treated as two separate loops.
    Therefore, the uncertainty of e.g. the saturation of one loop is not necessarily 
    considered for the other loop.
    This function is suitable for double hysteresis loops but not for single
    hysteretic loops due to overfitting.

    Parameters
    ----------
    xdata : list | pd.DataFrame | pd.Series | np.ndarray
        List of externally applied field strengths H.
    ydata : list | pd.DataFrame | pd.Series | np.ndarray
        List of magnetization values M.
    sat_cond : float, optional
        Saturation condition for the tanh function. Default is SAT_COND.
    param_estimates : dict, optional
        Initial guesses for the fit parameters. Default is None.
        'a': global offset, 
        'b_1': amplitude MS1 of the first loop,
        'c_1': steepness of the first loop,
        'd_1': global shift (HEB1) of the first loop,
        'e_1': branch offset (HC1) of the first loop,
        'b_2': amplitude MS2 of the second loop,
        'c_2': steepness of the second loop,
        'd_2': global shift (HEB2) of the second loop,
        'e_2': branch offset (HC2) of the second loop
    param_bounds : dict, optional
        Bounds for the fit parameters, has to be a dictionary of lists of length 2.
        The first element is the lower bound, the second element is the upper bound.
        Default is None.
    param_fixed : dict, optional
        Whether a parameter is fixed to its estimate (True) or not (False). 
        If param_estimates is not provided, the param is fixed to a numerical calculated estimate.
        Default is None.


    Returns
    -------
    tuple
        A tuple containing:
        - fitted_data : dict
            Dictionary containing:
            - xdata : pd.Series
                Interpolated xdata values.
            - ydata : pd.Series
                Fitted ydata values.
            - xdata_err : pd.Series
                Uncertainty of xdata values.
            - ydata_err : pd.Series
                Uncertainty of ydata values.
        - params : dict
            Dictionary containing:
            - r_squared : float
                Coefficient of determination of the fit.
            - HEB1 : float
                Exchange bias field strength for the first branch.
            - dHEB1 : float
                Uncertainty of exchange bias field for the first branch.
            - HC1 : float
                Coercive field strength for the first branch.
            - dHC1 : float
                Uncertainty of coercive field strength for the first branch.
            - HEB2 : float
                Exchange bias field strength for the second branch.
            - dHEB2 : float
                Uncertainty of exchange bias field for the second branch.
            - HC2 : float
                Coercive field strength for the second branch.
            - dHC2 : float
                Uncertainty of coercive field strength for the second branch.
            - MS1 : float
                Saturation magnetization for the first (left) branch.
            - dMS1 : float
                Uncertainty of saturation magnetization for the first (left) branch.
            - MS2 : float
                Saturation magnetization for the second (right) branch.
            - dMS2 : float
                Uncertainty of saturation magnetization for the second (right) branch.
            - MS1 : float
                Saturation magnetization for the first (left) branch.
            - dMS1 : float
                Uncertainty of saturation magnetization for the first (left) branch.
            - MS2 : float
                Saturation magnetization for the second (right) branch.
            - dMS2 : float
                Uncertainty of saturation magnetization for the second (right) branch.
            - MR : float
                Remanence at zero field strength.
            - dMR : float
                Uncertainty of remanence.
            - MHEB1 : float
                Magnetization at the exchange bias field for the first branch.
            - dMHEB1 : float
                Uncertainty of magnetization at the exchange bias field for the first branch.
            - MHEB2 : float
                Magnetization at the exchange bias field for the second branch.
            - dMHEB2 : float
                Uncertainty of magnetization at the exchange bias field for the second branch.
            - integral : float
                Area of the hysteresis loop.
            - dintegral : float
                Uncertainty of the area of the hysteresis loop.
            - integral1 : float
                Area of the first hysteresis loop.
            - dintegral1 : float
                Uncertainty of the area of the first hysteresis loop.
            - integral2 : float
                Area of the second hysteresis loop.
            - dintegral2 : float
                Uncertainty of the area of the second hysteresis loop.
            - saturation_fields1 : tuple
                Saturation field strengths for the first branch.
            - dsaturation_fields1 : tuple
                Uncertainty of saturation field strengths for the first branch.
            - saturation_fields2 : tuple
                Saturation field strengths for the second branch.
            - dsaturation_fields2 : tuple
                Uncertainty of saturation field strengths for the second branch.
            - slope_atHC1 : float
                Slope at the coercive field for the first branch.
            - dslope_atHC1 : float
                Uncertainty of the slope at the coercive field for the first branch.
            - slope_atHEB1 : float
                Slope at the exchange bias field for the first branch.
            - dslope_atHEB1 : float
                Uncertainty of the slope at the exchange bias field for the first branch.
            - slope_atHC2 : float
                Slope at the coercive field for the second branch.
            - dslope_atHC2 : float
                Uncertainty of the slope at the coercive field for the second branch.
            - slope_atHEB2 : float
                Slope at the exchange bias field for the second branch.
            - dslope_atHEB2 : float
                Uncertainty of the slope at the exchange bias field for the second branch.
            - alpha1 : float
                Angle enclosed between the slopes at HC and HEB for the first branch.
            - dalpha1 : float
                Uncertainty of the angle enclosed between the slopes at HC and HEB for the first branch.
            - rectangularity1 : float
                Rectangularity of the hysteresis loop for the first branch.
            - drectangularity1 : float
                Uncertainty of the rectangularity of the hysteresis loop for the first branch.
            - alpha2 : float
                Angle enclosed between the slopes at HC and HEB for the second branch.
            - dalpha2 : float
                Uncertainty of the angle enclosed between the slopes at HC and HEB for the second branch.
            - rectangularity2 : float
                Rectangularity of the hysteresis loop for the second branch.
            - drectangularity2 : float
                Uncertainty of the rectangularity of the hysteresis loop for the second branch.

    Raises
    ------
    TypeError
        If `xdata` or `ydata` is not of a supported type.
    ValueError
        If `xdata` and `ydata` do not have the same length.
    """
    _check_array_like(xdata, 'xdata')
    _check_array_like(ydata, 'ydata')

    x_data = np.asarray(xdata).copy()
    y_data = np.asarray(ydata).copy()
    # Check if the length of xdata is odd, i.e.the center point contributes to both branches.
    # Duplicate the center point in this case so that both branches are equally long.
    if len(x_data) % 2 != 0:
        center_index = len(x_data) // 2
        x_data = np.insert(x_data, center_index, x_data[center_index])
        y_data = np.insert(y_data, center_index, y_data[center_index])
    
    if len(x_data) != len(y_data):
        raise ValueError('xdata and ydata must have the same length')
    
    # # quick linear calculation to determine initial guesses for the exchange bias and the coercive field strength
    # LIN = lin_hyseval(x_data, y_data, sat_region=sat_region, use_offset=use_offset)
    # HEB_tmp, HC_tmp = LIN['HEB'], LIN['HC']
    
    # # Create a model from the function
    # model = Model(double_arctan_hys)

    # # Define the parameters
    # params = Parameters()
    # params.add('a', value=0.0) # offset, 0 for normalized hysteresis with pos/neg Sat.
    # params.add('b_1', value=(np.max(y_data) - np.min(y_data))/4) # amplitude, a quarter of the spread of the hysteresis, assuming that the two loops are of similar size
    # params.add('c_1', value=5.0) # steepness
    # params.add('d_1', value=HEB_tmp - 0.2 * np.abs(np.min(x_data)), min=np.min(x_data), max=np.max(x_data)) # lower exchange bias field
    # params.add('e_1', value=HC_tmp, min=-(np.max(x_data) - np.min(x_data)), max=np.max(x_data) - np.min(x_data)) # coercive field
    # params.add('b_2', value=(np.max(y_data) - np.min(y_data))/4) # amplitude, a quarter of the spread of the hysteresis
    # params.add('c_2', value=5.0) # steepness
    # # d_2 has to be larger or equal to d
    # params.add('d_2', value=HEB_tmp + 0.2 * np.abs(np.max(x_data)), min=np.min(x_data), max=np.max(x_data)) # upper exchange bias field
    # params.add('e_2', value=HC_tmp, min=-(np.max(x_data) - np.min(x_data)), max=np.max(x_data) - np.min(x_data)) # coercive field

        # locate each sub-loop's HEB/HC/amplitude/steepness directly from the branch separation,
    # instead of a single shared linear-fit guess offset by a fixed fraction for both sub-loops
    d_est, e_est, b_est, c_est = _estimate_multi_switching_params(x_data, y_data, n=2, sat_region=sat_region, use_offset=use_offset)

    # Create a model from the function
    model = Model(double_arctan_hys)

    # Define the parameters
    params = Parameters()
    params.add('a', value=0.0) # offset, 0 for normalized hysteresis with pos/neg Sat.
    params.add('b_1', value=b_est[0]) # amplitude, estimated from the first sub-loop's peak height
    params.add('c_1', value=c_est[0]) # steepness
    params.add('d_1', value=d_est[0], min=np.min(x_data), max=np.max(x_data)) # lower exchange bias field
    params.add('e_1', value=e_est[0], min=-(np.max(x_data) - np.min(x_data)), max=np.max(x_data) - np.min(x_data)) # coercive field
    params.add('b_2', value=b_est[1]) # amplitude, estimated from the second sub-loop's peak height
    params.add('c_2', value=c_est[1]) # steepness
    # d_2 has to be larger or equal to d
    params.add('d_2', value=d_est[1], min=np.min(x_data), max=np.max(x_data)) # upper exchange bias field
    params.add('e_2', value=e_est[1], min=-(np.max(x_data) - np.min(x_data)), max=np.max(x_data) - np.min(x_data)) # coercive field

    # Vary the parameters depending on additional input (normally not necessary)
    if param_estimates:
        for key, value in param_estimates.items():
            params[key].value = value
    if param_bounds:
        for key, value in param_bounds.items():
            params[key].min, params[key].max = value
    if param_fixed:
        for key, value in param_fixed.items():
            params[key].vary = not value 

    # Fit the model to the data
    result = model.fit(y_data, params, calc_covar=True, method=method, xdata=x_data,)

    # Calculate fitted magnetization values
    y_data_fitted = result.best_fit
    #convert to pd.Series for consistency
    y_data_fitted = pd.Series(y_data_fitted)

    # calculate the 3 sigma uncertainty of the fit
    y_data_fitted_err = result.eval_uncertainty(sigma=3)
    # convert to pd.Series for consistency
    y_data_fitted_err = pd.Series(y_data_fitted_err)

    # create a copy of the averaged xdata (branch 1 and 2) with which the results are
    # displayed
    # Split the array into two halves using slicing
    mid_index = len(x_data) // 2
    Xdata1 = x_data[:mid_index]
    Xdata2 = x_data[mid_index:]
    x_data_fitted = np.mean([Xdata2[::-1], Xdata1], axis=0)
    x_data_fitted = pd.Series([*x_data_fitted, *x_data_fitted[::-1]])

    #xdata uncertainty = half step size
    half_step_size = np.mean(np.abs(np.diff(x_data_fitted))) / 2
    #convert to pd.Series for consistency
    x_data_fitted_err = pd.Series([half_step_size] * len(x_data_fitted))

    fitted_data = {
        'xdata': x_data_fitted, 
        'ydata': y_data_fitted,
        'xdata_err': x_data_fitted_err,
        'ydata_err': y_data_fitted_err,
    }
    
    # tan_hyseval_params orders the fitted values by 1. increasing HEB and 2. by decreasing HC
    # So for the decreasing branch, the field numbers correspond to the switching behavior from 
    # left to the right. Unless for the unlikely but possible event a very large coercive field 
    # strength with a more higher EB field for the second loop
    params = arctan_hyseval_params(result, x_data, y_data, sat_cond=sat_cond)
    
    return fitted_data, params, result

def mult_arctan_hyseval(
    xdata: list | pd.DataFrame | pd.Series | np.ndarray, 
    ydata: list | pd.DataFrame | pd.Series | np.ndarray, 
    sat_cond: float = SAT_COND,
    sat_region: float = SAT_COND,
    use_offset: bool = True,
    n_arctan: int = 3, # for n = 1 or 2 the previous functions make more sense
    arctan_types: list | None = None,
    param_estimates: dict | None = None,
    param_bounds: dict | None = None,
    param_fixed: dict | None = None,
    method: str = 'leastsq',):

    """
    Fits a hysteresis loop with an arbitrarily choosen number of tanh function to extract several 
    parameters including the exchange bias field, coercive field strength, remanence, saturation
    magnetization, saturation field strength, the slopes at the intersection with the
    x-axis and the EB field, the steepness of the tanh function, a possible offset, the
    enclosed area and angle of the hysteresis loop, and the rectangularity of the hysteresis.

    It also provides the fitted ydata and its uncertainty as well as the lmfit result.

    Note: For the uncertainties, the hysteresis loops are treated as separate loops.
    Therefore, the uncertainty of e.g. the saturation of one loop is not considered for the 
    other loop, which may lead to a significant underestimation of the uncertainties for the individual loops.

    The number of arctan functions can either be choosen by the n_arctan parameter + param kwargs or in a 
    quick way by the arctan_types as e.g. ['EB', 'EB', 'FM', 'SP'] to define the hysts with constraints.

    Parameters
    ----------
    xdata : list or np.ndarray
        List of externally applied field strengths H.
    ydata : list or np.ndarray
        List of magnetization values M.
    sat_cond : float, optional
        Saturation condition for the tanh function. Default is SAT_COND.
    n_arctan : int, optional
        Number of arctan functions to be fitted. Default is 3.
    arctan_types : list, optional
        List of strings defining the type of each arctan function. Supported types are 'EB', 'FM', and 'SP'.
        If provided, the parameters of the arctan functions will be constrained according to their type.
        If arctan_types is provided, n_arctan is ignored. 
        The later estimates/bounds/fixes will overwrite the types if provided!
        Supported types are:
        - 'EB', 'exchange', 'exchange bias': no constraints for any param
        - 'FM', 'ferro', 'ferromagentic': no HEB field (parameter d needs to be 0)
        - 'SP', 'super', 'superpara', 'superparamagentic': HEB (d) and HC (e) need to be 0
        Default is None.
    param_estimates : dict, optional
        Initial guesses for the fit parameters. Default is None.
        'a': global offset, 
        'b_1': amplitude MS1 of the first loop,
        'c_1': steepness of the first loop,
        'd_1': global shift (HEB1) of the first loop,
        'e_1': branch offset (HC1) of the first loop,
        'b_2': amplitude MS2 of the second loop,
        'c_2': steepness of the second loop,
        'd_2': global shift (HEB2) of the second loop,
        'e_2': branch offset (HC2) of the second loop
        ...
    param_bounds : dict, optional
        Bounds for the fit parameters, has to be a dictionary of lists of length 2 with the same
        keys as param_estimates. The first element is the lower bound, the second element is the 
        upper bound.
        Default is None.
    param_fixed : dict, optional
        Whether a parameter is fixed to its estimate (True) or not (False). 
        If param_estimates is not provided, the param is fixed to a numerical calculated estimate.
        Default is None.


    Returns
    -------
    fitted_data : np.ndarray

    Raises
    ------
    ValueError
        If `xdata` and `ydata` do not have the same length.
        If `xdata` or `ydata` is not of a supported type.
    """
    if arctan_types is None:
            arctan_types = []
    _check_array_like(xdata, 'xdata')
    _check_array_like(ydata, 'ydata')
    
    x_data = np.asarray(xdata).copy()
    y_data = np.asarray(ydata).copy()
    # Check if the length of xdata is odd, i.e.the center point contributes to both branches.
    # Duplicate the center point in this case so that both branches are equally long.
    if len(x_data) % 2 != 0:
        center_index = len(x_data) // 2
        x_data = np.insert(x_data, center_index, x_data[center_index])
        y_data = np.insert(y_data, center_index, y_data[center_index])
    
    if len(x_data) != len(y_data):
        raise ValueError('xdata and ydata must have the same length')

    # # quick linear calculation to determine initial guesses for the exchange bias and the coercive field strength
    # LIN = lin_hyseval(x_data, y_data, sat_region=sat_region, use_offset=use_offset)
    # HEB_tmp, HC_tmp = LIN['HEB'], LIN['HC']
    # locate each sub-loop's HEB/HC/amplitude/steepness directly from the branch separation,
    # instead of a single shared linear-fit guess offset by a small fixed fraction per sub-loop
    n_estimates = n_arctan if not arctan_types else len(arctan_types)
    d_est, e_est, b_est, c_est = _estimate_multi_switching_params(x_data, y_data, n=n_estimates, sat_region=sat_region, use_offset=use_offset)

    # Define the parameters
    params = Parameters()
    params.add('a', value=np.mean(y_data)) # offset, assume mean of the data for better convergence
    if arctan_types:
        for n in range(1, n_arctan + 1):
            # params.add(f'b_{n}', value=(np.max(y_data) - np.min(y_data))/(2*n_arctan)) # amplitude, equal portion of the hyst's mag
            # params.add(f'c_{n}', value=5.0, min=0) # steepness
            # params.add(f'd_{n}', value=HEB_tmp + 0.02 * np.abs(np.min(x_data)) * n, min=np.min(x_data), max=np.max(x_data)) # lower exchange bias field
            # params.add(f'e_{n}', value=HC_tmp, min=-(np.max(x_data) - np.min(x_data)), max=np.max(x_data) - np.min(x_data)) # coercive field
            params.add(f'b_{n}', value=b_est[n - 1]) # amplitude, estimated from this sub-loop's peak height
            params.add(f'c_{n}', value=c_est[n - 1], min=0) # steepness
            params.add(f'd_{n}', value=d_est[n - 1], min=np.min(x_data), max=np.max(x_data)) # exchange bias field
            params.add(f'e_{n}', value=e_est[n - 1], min=-(np.max(x_data) - np.min(x_data)), max=np.max(x_data) - np.min(x_data)) # coercive field

    elif isinstance(arctan_types, list):
        # check if all entries are supported:
        for arctan_type, n in zip(arctan_types, range(1, len(arctan_types) + 1)):
            if arctan_type.lower() in ['eb', 'exchange', 'exchange bias']:
                arctan_type = 'eb'
            elif arctan_type.lower() in ['fm', 'ferro', 'ferromagnetic']:
                arctan_type = 'fm'
            elif arctan_type.lower() in ['sp', 'super', 'superpara', 'superparamagnetic']:
                arctan_type = 'sp'
            else:
                raise ValueError(f"The arctan_type {arctan_type} is not supported. Please check the provided arctan_types list: {arctan_types}.")

            # params.add(f'b_{n}', value=(np.max(y_data) - np.min(y_data))/(2*n_arctan)) # amplitude, equal portion of the hyst's mag
            # params.add(f'c_{n}', value=5.0) # steepness
            # params.add(f'd_{n}', value=HEB_tmp + 0.02 * np.abs(np.min(x_data)) * n, min=np.min(x_data), max=np.max(x_data)) # lower exchange bias field
            # params.add(f'e_{n}', value=HC_tmp, min=-(np.max(x_data) - np.min(x_data)), max=np.max(x_data) - np.min(x_data)) # coercive field
            params.add(f'b_{n}', value=b_est[n - 1]) # amplitude, estimated from this sub-loop's peak height
            params.add(f'c_{n}', value=c_est[n - 1]) # steepness
            params.add(f'd_{n}', value=d_est[n - 1], min=np.min(x_data), max=np.max(x_data)) # exchange bias field
            params.add(f'e_{n}', value=e_est[n - 1], min=-(np.max(x_data) - np.min(x_data)), max=np.max(x_data) - np.min(x_data)) # coercive field

            # if arctan_types = 'fm' change HEB (d_n) to 0 and put vary=False
            # Same for HEB (d_n) AND HC (e_n) if it is 'sp'
            if arctan_type == 'fm':
                params[f'd_{n}'].value = 0
                params[f'd_{n}'].vary = False
            if arctan_type == 'sp':
                params[f'd_{n}'].value = 0
                params[f'd_{n}'].vary = False
                params[f'e_{n}'].value = 0
                params[f'e_{n}'].vary = False

    else:
        raise TypeError(f"arctan_types must be a list or None, not {type(arctan_types)}")

    # Vary the parameters depending on additional input (normally not necessary, will overwrite tan_types)
    if param_estimates:
        for key, value in param_estimates.items():
            params[key].value = value
    if param_bounds:
        for key, value in param_bounds.items():
            params[key].min, params[key].max = value
    if param_fixed:
        for key, value in param_fixed.items():
            params[key].vary = not value 

     # Create a model from the function
    model = Model(mult_arctan_hys, independent_vars=['xdata'])

    # Fit the model to the data
    result = model.fit(y_data, params, calc_covar=True, method=method, xdata=x_data,)

    # Calculate fitted magnetization values
    y_data_fitted = result.best_fit
    #convert to pd.Series for consistency
    y_data_fitted = pd.Series(y_data_fitted)

    # calculate the 3 sigma uncertainty of the fit
    y_data_fitted_err = result.eval_uncertainty(sigma=3)
    # convert to pd.Series for consistency
    y_data_fitted_err = pd.Series(y_data_fitted_err)

    # create a copy of the averaged xdata (branch 1 and 2) with which the results are
    # displayed
    # Split the array into two halves using slicing
    mid_index = len(x_data) // 2
    Xdata1 = x_data[:mid_index]
    Xdata2 = x_data[mid_index:]
    x_data_fitted = np.mean([Xdata2[::-1], Xdata1], axis=0)
    x_data_fitted = pd.Series([*x_data_fitted, *x_data_fitted[::-1]])

    #xdata uncertainty = half step size
    half_step_size = np.mean(np.abs(np.diff(x_data_fitted))) / 2
    #convert to pd.Series for consistency
    x_data_fitted_err = pd.Series([half_step_size] * len(x_data_fitted))

    fitted_data = {
        'xdata': x_data_fitted, 
        'ydata': y_data_fitted,
        'xdata_err': x_data_fitted_err,
        'ydata_err': y_data_fitted_err,
    }

    # tan_hyseval_params orders the fitted values by 1. increasing HEB and 2. by decreasing HC
    # So for the decreasing branch, the field numbers correspond to the switching behavior from 
    # left to the right. Unless for the unlikely but possible event a very large coercive field 
    # strength with a more higher EB field for the second loop
    params = arctan_hyseval_params(result, x_data, y_data, sat_cond)

    return fitted_data, params, result

def arctan_hyseval_params(
    result, 
    xdata: list | pd.DataFrame | pd.Series | np.ndarray, 
    ydata: list | pd.DataFrame | pd.Series | np.ndarray, 
    sat_cond = SAT_COND):
    """
    Analytical calculation of important parameters of a hysteresis loop fitted with a tanh function
    (see tan_hyseval and tan_hys for details).
    
    Parameters
    ----------
    result : lmfit.model.ModelResult
        Result of the fit of the tanh function to the hysteresis loop.
    xdata : list | pd.DataFrame | pd.Series | np.ndarray
        List of externally applied field strengths H.
    ydata : list | pd.DataFrame | pd.Series | np.ndarray
        List of magnetization values M.
    sat_cond : float, optional
        Saturation condition for the tanh function. Default is SAT_COND.
    
    Returns
    -------
    dict
        Dictionary containing:
        - HEB : float
            Exchange bias field strength.
        - HC : float
            Coercive field strength.
        - MS : float
            Saturation magnetization.
        - MR : float
            Remanence.
        - MHEB : float
            Magnetization at the exchange bias field.
        - saturation_fields : tuple
            Saturation field strengths.
        - slope_atHC : float
            Slope at the coercive field.
        - slope_atHEB : float
            Slope at the exchange bias field.
        - alpha : float
            Angle enclosed between the slopes at HC and HEB.
        - rectangularity : float
            Rectangularity of the hysteresis loop.
            
        For every quantity, the uncertainty is also calculated (error propagation), 
        which then has the prefix 'd'.
        For the saturation field strengths, the uncertainty is a scalar value and true for both fields.
        For MR and MHEB, the uncertainty is also a scalar value, true for all three values (abs. average, decreasing, increasing).
        
    Raises
    ------
    TypeError
        If `xdata` or `ydata` is not of a supported type.
    ValueError
        If `xdata` and `ydata` do not have the same length or if the number of parameters in the result is not correct.
    """
    _check_array_like(xdata, 'xdata')
    _check_array_like(ydata, 'ydata')
    
    x_data = np.asarray(xdata).copy()
    y_data = np.asarray(ydata).copy()
    
    if len(x_data) != len(y_data) or len(x_data)%2 != 0:
        raise ValueError('xdata and ydata must have the same length and be even, i.e. the center point contributes to both branches.')
    
    # Check the length of the params dictionary. The following calculation should be performed n times for every 4*n + 1 parameters.
    if len(result.params) % 4 != 1:
        raise ValueError("The parameter dictionary does not contain the correct number of parameters.")
    
    nr_of_calculations = (len(result.params) - 1) // 4
    
    #xdata uncertainty = half step size
    half_step_size = np.mean(np.abs(np.diff(x_data))) / 2
    
    params = {}
    
    for i in range(1, nr_of_calculations + 1):
        # Assign the parameters a, b, c, d, e to the corresponding values of the current calculation
        a = result.params['a'].value # offset, equal for all calculations
        a_err = _safe_stderr(result.params['a'].stderr)
        
        b = result.params[f'b_{i}'].value # amplitude
        b_err = _safe_stderr(result.params[f'b_{i}'].stderr)
        
        c = result.params[f'c_{i}'].value # steepness
        c_err = _safe_stderr(result.params[f'c_{i}'].stderr)
        
        d = result.params[f'd_{i}'].value # exchange bias field
        d_err = _safe_stderr(result.params[f'd_{i}'].stderr)

        # TODO: check if np.abs() makes sense here or later
        e = np.abs(result.params[f'e_{i}'].value) # coercive field
        e_err = _safe_stderr(result.params[f'e_{i}'].stderr)
        
        # Extract wanted values from the fits optimized params
        HC, dHC = e, e_err + half_step_size
        HEB, dHEB = d, d_err + half_step_size
        MS, dMS = b, b_err + a_err
        MR = arctan_hys(float(0), a, b, c, d, e)
        dMR = a_err + b_err # TODO: check if this is correct
        MHEB = arctan_hys(float(HEB), a, b, c, d, e)
        dMHEB = dMR # TODO: check if this is correct

        slope_atHC = b * c
        dslope_atHC = b_err * c + c_err * b
        
        slope_atHEB = (b * c) / (1 + (c * e)**2)
        # partial differentiation for the error calculation
        dslope_atHEB_db = (c + (c * e)**2 ) * b_err
        dslope_atHEB_dc = b * c_err
        dslope_atHEB_de = 2 * b * c**2 * e * e_err
        dslope_atHEB = (dslope_atHEB_db + dslope_atHEB_dc + dslope_atHEB_de) / (1 + (c * e)**2)**2
        
        slope_atMR = (b * c) / (1 + (c * d)**2)
        # dslope_atMR_db = (c + (c * d)**2 ) * b_err
        # dslope_atMR_dc = b * c_err
        # dslope_atMR_dd = 2 * b * c**2 * d * d_err
        # dslope_atMR = (dslope_atMR_db + dslope_atMR_dc + dslope_atMR_dd) / (1 + (c * d)**2)**2
        
        dMHEB = dMS + slope_atHEB * dHEB
        dMR = dMS + slope_atMR * half_step_size

        # Calculate the rectangularity of the hysteresis loop by interpolating a parallelogram with the slopes at HC and HEB. The closer these angles are to 90°, the more rectangular the hysteresis loop is.
        # Normalize the slope to make it independent on the scaling for MS and Hext: divide 2 * HC by the amplitude of the hysteresis loop
        slope_normalization = 2 * HC / b

        # angles of the slopes at HC and HEB
        angle_atHC = np.arctan(slope_atHC * slope_normalization)
        dangle_atHC = slope_normalization / (1 + slope_atHC**2 * slope_normalization**2) * dslope_atHC
        angle_atHEB = np.arctan(slope_atHEB * slope_normalization)
        dangle_atHEB = slope_normalization / (1 + slope_atHEB**2 * slope_normalization**2) * dslope_atHEB
        
        # angles at the intersection points. Alpha1 is the angle enclosed between the slopes at HC and HEB, i.e. the angle at the lower left corner of the parallelogram. Alpha2 is the angle at the upper right corner but their sum is always 180°, so it does not need to be calculated.
        alpha = angle_atHC - angle_atHEB
        dalpha = np.sqrt(dangle_atHC**2 + dangle_atHEB**2)
        rectangularity = np.sin(alpha) # from 0 to 1, 1 is a perfect rectangle at alpha1 = 90°
        drectangularity = np.abs(np.cos(alpha)) * dalpha

        # calculate saturation field strength (at SAT_COND*max(arctan(x))), ignore a as it is a global offset and does not influence the max of the spread
        # f(x) = a + b * 2/np.pi * np.arctan(c * (x - d +- e)); a = 0
        # arctan approaches maximum value of pi/2 so max of def arctan 
        # is a + b * 2/np.pi * np.pi/2 = a + b = b for a = 0
        # so SAT_COND * +- b = b * 2/np.pi * arctan ( c * (x_sat - d +- e))
        # +- SAT_COND * b = b * 2/np.pi * arctan ( c * (x_sat - d +- e))
        # +- SAT_COND = 2/np.pi * arctan ( c * (x_sat - d +- e))
        # np.pi/2 * ( +- SAT_COND) = arctan ( c * (x_sat - d +- e))

        # tan(np.pi/2 * ( +- SAT_COND)) = c * (x_sat - d +- e)
        # tan(np.pi/2 * ( +- SAT_COND)) / c = x_sat - d +- e
        
        # tan(np.pi/2 * ( +- SAT_COND)) / c + d -+ e = x_sat

        # descending branch (neg. saturation field, positive HC shift): x_sat_desc = tan(np.pi/2 * ( - SAT_COND)) / c + d + e
        # ascending branch (pos. saturation field, negative HC shift): x_sat_asc = tan(np.pi/2 * ( + SAT_COND)) / c + d - e

        # x_sat error is equal for both branches
        
        # dx_sat_dc = np.abs( -tan(np.pi/2 * ( +- SAT_COND)) / c**2 * c_err )
        # dx_sat_de = np.abs(1 * e_err)
        # dx_sat_dd = np.abs(1 * d_err)
        # dx_sat = dx_sat_dc + d_err + e_err + half_step_size
        
        saturation_field_desc = np.tan(np.pi / 2 * (-sat_cond)) / c + HEB + HC
        saturation_field_asc = np.tan(np.pi / 2 * sat_cond) / c + HEB - HC
        saturation_fields = (saturation_field_desc, saturation_field_asc)
        # error calculation
        dsaturation_fields_dc = -np.tan(sat_cond * np.pi / 2 - (1-sat_cond) * a / b) / c**2 * c_err
        dsaturation_fields = np.abs(dsaturation_fields_dc) + d_err + e_err + half_step_size

        integral_args = {
            'a': a,
            'b': b,
            'c': c,
            'd': d,
            'e': e,
        }
        # change param a and e
        integral_args['a'] -= np.min(ydata) # change offset so that the integral starts at 0
        # Integral of the area in the loop = left branch integral - right branch integral
        integral_leftbranch, integral_leftbranch_err = quad(arctan, np.min(xdata), np.max(xdata), args=tuple(integral_args.values()))
        integral_args['e'] *= -1
        integral_rightbranch, integral_rightbranch_err = quad(arctan, np.min(xdata), np.max(xdata), args=tuple(integral_args.values()))
        # Area of the hysteresis loop
        integral = integral_leftbranch - integral_rightbranch
        integral_err = np.sqrt(integral_leftbranch_err**2 + integral_rightbranch_err**2)

        hys_key = f'_{i}'

        params.update({
            
            'HEB' + hys_key: HEB,
            'dHEB' + hys_key: dHEB,
            'HC' + hys_key: HC,
            'dHC' + hys_key: dHC,
            'MS' + hys_key: MS,
            'dMS' + hys_key: dMS,
            'MR' + hys_key: MR,
            'dMR' + hys_key: dMR,
            'MHEB' + hys_key: MHEB,
            'dMHEB' + hys_key: dMHEB,
            'saturation_fields' + hys_key: saturation_fields,
            'dsaturation_fields' + hys_key: dsaturation_fields,
            'slope_atHC' + hys_key: slope_atHC,
            'dslope_atHC' + hys_key: dslope_atHC,
            'slope_atHEB' + hys_key: slope_atHEB,
            'dslope_atHEB' + hys_key: dslope_atHEB,
            'alpha' + hys_key: alpha,
            'dalpha' + hys_key: dalpha,
            'rectangularity' + hys_key: rectangularity,
            'drectangularity' + hys_key: drectangularity,
            
            'integral' + hys_key: integral,
            'dintegral' + hys_key: integral_err,
        })
        
    # add the result.param values and stderrs as well as the r_squared value and the chi_squared value
    for key, value in result.params.valuesdict().items():
        params[key] = value
        params['d' + key] = _safe_stderr(result.params[key].stderr)
        
    if nr_of_calculations > 1:
        # Sort the params dictionary by 1. increasing HEB field and 2. decreasing coercive field strength
        HEBs = [params['HEB_' + str(i)] for i in range(1, nr_of_calculations + 1)]
        HCs = [params['HC_' + str(i)] for i in range(1, nr_of_calculations + 1)]
        sorted_indices = np.lexsort(([-hc for hc in HCs], HEBs)) # sort by HEB ascending and then by HC descending
        sorted_params = {}
        for key in params:  # noqa: PLC0206
            if '_' in key:
                base_key, index = key.rsplit('_', 1)
                if index.isdigit():
                    new_index = sorted_indices[int(index)-1] + 1
                    sorted_params[base_key + '_' + str(new_index)] = params[key]
                else:
                    sorted_params[key] = params[key]
            else:
                sorted_params[key] = params[key]
        params = sorted_params

    params['r_squared'] = result.rsquared
    params['chi_squared'] = result.chisqr

    return params

def batch_hyseval(
    datasets: dict,
    eval_func=arctan_hyseval,
    eval_props: dict | None = None,
    prep: bool = True,
    prep_props: dict | None = None,
):
    """
    Run a hysteresis evaluation function on many loops at once and collect
    the fit parameters into a single table.

    Convenience wrapper for the common case of evaluating many measurement
    files with the same settings, e.g. comparing a batch of samples or
    scanning a parameter (temperature, layer thickness, etc.).

    Parameters
    ----------
    datasets : dict
        Dictionary mapping a label (e.g. sample name or filename) to a
        ``(xdata, ydata)`` tuple of the raw hysteresis loop.
    eval_func : callable, optional
        Evaluation function applied to every loop, e.g. `arctan_hyseval`,
        `double_arctan_hyseval`, `mult_arctan_hyseval`, or `lin_hyseval`.
        Default is `arctan_hyseval`.
    eval_props : dict, optional
        Keyword arguments passed on to `eval_func` for every loop.
    prep : bool, optional
        If True (default), `hys_prep` is applied to every loop before fitting.
    prep_props : dict, optional
        Keyword arguments passed on to `hys_prep`.

    Returns
    -------
    params_table : pandas.DataFrame
        One row per successfully evaluated loop, with a 'label' column and
        one column per fit parameter.
    fitted_curves : dict
        Maps each label to its fitted_data dict (xdata, ydata, xdata_err,
        ydata_err), for evaluation functions that return one (everything
        except `lin_hyseval`).
    failed : dict
        Maps the label of every loop that could not be evaluated to the
        error message, so problem files can be inspected individually
        instead of stopping the whole batch.

    Examples
    --------
    >>> datasets = {'sample_A': (H_A, M_A), 'sample_B': (H_B, M_B)}
    >>> table, curves, failed = batch_hyseval(datasets)
    """
    if not isinstance(datasets, dict):
        raise TypeError(f'datasets must be a dict of label: (xdata, ydata) pairs, not {type(datasets)}')

    eval_props = eval_props or {}
    prep_props = prep_props or {}

    rows = []
    fitted_curves = {}
    failed = {}

    for label, (xdata, ydata) in datasets.items():
        try:
            x, y = hys_prep(xdata, ydata, **prep_props) if prep else (xdata, ydata)
            output = eval_func(x, y, **eval_props)

            # eval_func returns either (fitted_data, params, result), or just
            # a params dict (lin_hyseval)
            if isinstance(output, tuple):
                fitted_data, params, _result = output
                fitted_curves[label] = fitted_data
            else:
                params = output

            rows.append({'label': label, **params})

        except Exception as error:  # keep going: one bad loop shouldn't stop the batch
            failed[label] = str(error)

    params_table = pd.DataFrame(rows)
    if not params_table.empty:
        params_table = params_table[['label'] + [c for c in params_table.columns if c != 'label']]

    if failed:
        print(f'{len(failed)} of {len(datasets)} loop(s) could not be evaluated: {list(failed.keys())}')

    return params_table, fitted_curves, failed


#%%
###############################################################################
# 6. Further functions, mainly for export, conversion and plotting
###############################################################################

def export_results(params_table: pd.DataFrame, path: str, fitted_curves: dict | None = None):
    """
    Save batch evaluation results to a CSV or Excel file.

    Parameters
    ----------
    params_table : pandas.DataFrame
        Table of fit parameters, e.g. as returned by `batch_hyseval`.
    path : str
        Output file path. The extension selects the format:
        '.csv' for a comma-separated file, '.xlsx' for an Excel workbook.
    fitted_curves : dict, optional
        Maps labels to fitted_data dicts, as returned by `batch_hyseval`.
        Only used for '.xlsx' export: each loop's fitted curve is written
        to its own worksheet alongside a 'results' worksheet.

    Raises
    ------
    ValueError
        If the file extension is not '.csv' or '.xlsx'.

    Examples
    --------
    >>> export_results(params_table, 'results.csv')
    >>> export_results(params_table, 'results.xlsx', fitted_curves)
    """
    path = str(path)

    if path.endswith('.csv'):
        # Columns holding tuples (e.g. 'MR', 'saturation_fields') are written as their
        # string representation in a .csv file. Use .xlsx export to keep them numeric.
        params_table.to_csv(path, index=False)

    elif path.endswith('.xlsx'):
        with pd.ExcelWriter(path) as writer:  # requires the 'openpyxl' package
            params_table.to_excel(writer, sheet_name='results', index=False)
            if fitted_curves:
                for label, fitted_data in fitted_curves.items():
                    sheet_name = str(label)[:31]  # Excel sheet names are capped at 31 characters
                    pd.DataFrame(fitted_data).to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        raise ValueError(f"Unsupported file extension for '{path}'. Use '.csv' or '.xlsx'.")

    print(f'Results saved to {path}')

def branch_recognition(xdata: pd.DataFrame, ydata: pd.DataFrame):
    """
    This function recognizes which data points belong to the ascending and which to the descending branch of a hysteresis loop.
    Then it will sort the data points accordingly and return the two branches individually.
    The descending branch has a strictly decreasing x-value, while the ascending branch has a strictly increasing x-value.
    """
    
    # Look at the difference in the x-values
    diff_x = np.diff(xdata)
    branch_turning_points = [0]
    
    # Check for 0 values or sign changes
    for i, diff in enumerate(diff_x[:-1]):
        if diff * diff_x[i+1] <= 0:
            # Add the index of the turning point
            branch_turning_points.append(i)
    
    # Calculate the sum of the diff_x values between the turning points
    sum_diff_x = [np.sum(diff_x[branch_turning_points[i]:branch_turning_points[i+1]]) for i in range(len(branch_turning_points))]
    
    # For now, assume there are only two branches. TODO: Implement for more than two branches
    # negative sum_diff_x means descending branch, positive sum_diff_x means ascending branch
    if sum_diff_x[0] < 0:
        # descending branch first
        xdata_new = [xdata.iloc[branch_turning_points[0]:branch_turning_points[1]+1], xdata.iloc[branch_turning_points[1]:]]
        ydata_new = [ydata.iloc[branch_turning_points[0]:branch_turning_points[1]+1], ydata.iloc[branch_turning_points[1]:]]
    elif sum_diff_x[0] > 0:
        # ascending branch first
        xdata_new = [xdata.iloc[branch_turning_points[1]:], xdata.iloc[branch_turning_points[0]:branch_turning_points[1]+1]]
        ydata_new = [ydata.iloc[branch_turning_points[1]:], ydata.iloc[branch_turning_points[0]:branch_turning_points[1]+1]]
        
    else:
        raise ValueError("The data does not seem to contain a valid hysteresis loop. Please check the input data.")
    
    return xdata_new, ydata_new

def create_uncertainty_polygon(xdata: pd.DataFrame, ydata: pd.DataFrame, xdata_err: pd.DataFrame, ydata_err: pd.DataFrame):
    """
    Create a polygon that covers all uncertainties in x and y directions.

    This function assumes both uncertainties to be independent/uncorrelated,
    which may underestimate the actual uncertainty. In the case of hysteresis loops,
    the polygon is used to calculate the uncertainty band/polygon of just one branch,
    as overlapping drawings in the saturation region may cause problems in the visualization.

    Parameters
    ----------
    xdata : pd.DataFrame
        DataFrame of externally applied field strengths H, typically of a single branch.
    ydata : pd.DataFrame
        DataFrame of magnetization values M, typically of a single branch.
    xdata_err : pd.DataFrame
        DataFrame of uncertainties in the externally applied field strengths H.
    ydata_err : pd.DataFrame
        DataFrame of uncertainties in the magnetization values M.

    Returns
    -------
    polygon_x : np.ndarray
        Array of x-coordinates of the polygon vertices.
    polygon_y : np.ndarray
        Array of y-coordinates of the polygon vertices.

    Examples
    --------
    >>> import pandas as pd
    >>> xdata = pd.DataFrame([1, 2, 3])
    >>> ydata = pd.DataFrame([4, 5, 6])
    >>> xdata_err = pd.DataFrame([0.1, 0.2, 0.1])
    >>> ydata_err = pd.DataFrame([0.2, 0.1, 0.2])
    >>> create_uncertainty_polygon(xdata, ydata, xdata_err, ydata_err)
    (array([1.1, 2.2, 3.1, 2.9, 1.8, 0.9]), array([4.2, 5.1, 6.2, 5.8, 4.9, 3.8]))
    """
    # Ensure inputs are numpy arrays for easier manipulation
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    xdata_err = np.array(xdata_err)
    ydata_err = np.array(ydata_err)

    # Calculate the upper and lower bounds for x and y
    x_upper = xdata + xdata_err
    x_lower = xdata - xdata_err
    y_upper = ydata + ydata_err
    y_lower = ydata - ydata_err

    # Create the polygon vertices
    polygon_x = np.concatenate([x_upper, x_lower[::-1]])
    polygon_y = np.concatenate([y_upper, y_lower[::-1]])

    return polygon_x, polygon_y

def plot_loop(
    xdata, 
    ydata, 
    xdata_err=None, 
    ydata_err=None,
    xunit=None,
    yunit=None,
    ax=None,
    plotstyle='line', #scatter, line or both
    label=None,
    title=None, 
    color=None,
    vmin=0,
    vmax=1,
    line_kwargs=None,
    scatter_kwargs=None,
    fill_kwargs=None,
    legend_kwargs=None,
    grid_kwargs=None,
    show_legend=True,):
    """
    Plot one hysteresis loop with readable defaults and branch-aware styling.

    The loop is split into increasing and decreasing field branches based on
    ``np.diff(xdata)`` and visualized separately. This allows branch-specific
    coloring and uncertainty polygons while keeping a compact API.

    Compared to a plain matplotlib call, this helper adds:
    - automatic branch detection,
    - optional uncertainty polygons,
    - optional colormap progression along the loop,
    - sensible style defaults that can be overridden via ``*_kwargs``.

    Parameters
    ----------
    xdata, ydata : list | pd.DataFrame | pd.Series | np.ndarray
        Hysteresis data points.
    xdata_err, ydata_err : list | pd.DataFrame | pd.Series | np.ndarray, optional
        Pointwise uncertainties for ``xdata`` and ``ydata``. If both are given,
        branch-wise uncertainty polygons are plotted.
    xunit, yunit : str, optional
        Unit strings used in axis labels.
    ax : matplotlib.axes.Axes, optional
        Axis object to draw on. If ``None``, the current axis is used.
    plotstyle : str, optional
        One of ``'line'``, ``'scatter'``, or ``'both'``.
    label : str, optional
        Legend label for the increasing branch.
    title : str, optional
        Plot title.
    color : str | tuple | list | matplotlib.colors.Colormap, optional
        Color handling:
        - str: same color for both branches,
        - 2-item tuple/list: ``(increasing, decreasing)``,
        - Colormap: gradient along the loop,
        - None: defaults to blue shades.
    vmin, vmax : float, optional
        Normalization range used when ``color`` is a colormap.
    line_kwargs : dict, optional
        Extra keyword arguments forwarded to ``ax.plot`` and ``colored_line``.
        Entries here override the function defaults.
    scatter_kwargs : dict, optional
        Extra keyword arguments forwarded to ``ax.scatter``.
        Entries here override the function defaults.
    fill_kwargs : dict, optional
        Extra keyword arguments forwarded to ``ax.fill`` for uncertainty bands.
        Entries here override the function defaults.
    legend_kwargs : dict, optional
        Extra keyword arguments forwarded to ``ax.legend``.
    grid_kwargs : dict, optional
        Extra keyword arguments forwarded to ``ax.grid``.
    show_legend : bool, optional
        If ``True``, draw a legend only when labeled artists exist.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the plot.

    Raises
    ------
    ValueError
        If data lengths do not match or ``plotstyle`` is invalid.
    """
    if ax is None:
        ax = plt.gca()

    line_kwargs = {} if line_kwargs is None else dict(line_kwargs)
    scatter_kwargs = {} if scatter_kwargs is None else dict(scatter_kwargs)
    fill_kwargs = {} if fill_kwargs is None else dict(fill_kwargs)
    legend_kwargs = {} if legend_kwargs is None else dict(legend_kwargs)
    grid_kwargs = {} if grid_kwargs is None else dict(grid_kwargs)

    # Readable defaults while preserving full user override capability.
    line_defaults = {'linewidth': 2.0, 'zorder': 3}
    scatter_defaults = {'s': 24, 'zorder': 4, 'alpha': 0.9}
    fill_defaults = {'alpha': 0.18, 'zorder': 2}
    grid_defaults = {'ls': '--', 'alpha': 0.35, 'zorder': 0}

    line_style = {**line_defaults, **line_kwargs}
    scatter_style = {**scatter_defaults, **scatter_kwargs}
    fill_style = {**fill_defaults, **fill_kwargs}
    grid_style = {**grid_defaults, **grid_kwargs}

    # Accept pandas and numpy inputs by normalizing to 1D arrays.
    x = np.asarray(xdata).ravel()
    y = np.asarray(ydata).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError("xdata and ydata must have the same length.")
    
    # split the data into two branches, one for increasing x and one for decreasing x
    # Handle points where diff == 0 (turnaround points, initial point) by forward-filling the branch sign
    if len(x) > 1:
        diffs = np.diff(x)
        signs = np.sign(diffs)
        # Prepend the sign of the first difference for the first point
        signs = np.concatenate([[signs[0] if len(signs) > 0 else 1], signs])
        # Forward-fill zeros: assign zero-diff points to the branch of the next point, which is mainly important at the switching point between the branches.
        for i in range(1, len(signs)-1):
            if signs[i] == 0:
                signs[i] = signs[i + 1]
        increasing = signs >= 0
    else:
        increasing = np.ones(len(x), dtype=bool)
    xdata_inc = x[increasing]
    ydata_inc = y[increasing]
    xdata_dec = x[~increasing]
    ydata_dec = y[~increasing]
    
    # if a single color is provided, use it for both branches
    if isinstance(color, colors.Colormap):
        color_inc = color(0.25)
        color_dec = color(0.75)
    elif isinstance(color, str):
        color_inc = color
        color_dec = color
    # for two colors, use the first for the increasing branch and the second for the decreasing branch
    elif isinstance(color, (list, tuple)) and len(color) == 2:
        color_inc = color[0]
        color_dec = color[1]
    # if no color is provided, use blue for the increasing branch and dark blue for the decreasing branch
    else:
        color_inc = 'blue'
        color_dec = 'darkblue'
    # 4th case of a cmap is handled later with the function colored_line

    if xdata_err is not None and ydata_err is not None:
        xerr = np.asarray(xdata_err).ravel()
        yerr = np.asarray(ydata_err).ravel()
        if xerr.shape[0] != x.shape[0] or yerr.shape[0] != y.shape[0]:
            raise ValueError("xdata_err and ydata_err must match xdata and ydata length.")

        xdata_err_inc = xerr[increasing]
        ydata_err_inc = yerr[increasing]
        xdata_err_dec = xerr[~increasing]
        ydata_err_dec = yerr[~increasing]
        
        # plot the uncertainty polygons for both branches
        polygon_x_inc, polygon_y_inc = create_uncertainty_polygon(xdata_inc, ydata_inc, xdata_err_inc, ydata_err_inc)
        polygon_x_dec, polygon_y_dec = create_uncertainty_polygon(xdata_dec, ydata_dec, xdata_err_dec, ydata_err_dec)
        ax.fill(polygon_x_inc, polygon_y_inc, color=color_inc, **fill_style)
        ax.fill(polygon_x_dec, polygon_y_dec, color=color_dec, **fill_style)
        
    if isinstance(color, colors.Colormap):
        # Map color values for both branches using ScalarMappable for proper colorbar support
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        # scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=color)
        # check if the data starts with an increasing branch to correctly assign the cmap gradient
        start_branch = 'increasing' if increasing[0] else 'decreasing'
            
        if plotstyle == 'line' or plotstyle == 'both':
            if start_branch == 'increasing':
                c_inc = np.linspace(0, 0.5, max(len(xdata_inc) - 1, 1))
                c_dec = np.linspace(0.5, 1, max(len(xdata_dec) - 1, 1))
            else:
                c_inc = np.linspace(0.5, 1, max(len(xdata_inc) - 1, 1))
                c_dec = np.linspace(0, 0.5, max(len(xdata_dec) - 1, 1))
            
            # Plot lines with color gradient
            if len(xdata_inc) > 0:
                if label is not None:
                    _label = label
                elif plotstyle == 'line':
                    _label = 'hysteresis loop'
                else:
                    _label = None
                lc_inc = colored_line(xdata_inc, ydata_inc, c_inc, ax, cmap=color, label=_label, **line_style)
                lc_inc.set_norm(norm)
            if len(xdata_dec) > 0:
                lc_dec = colored_line(xdata_dec, ydata_dec, c_dec, ax, cmap=color, **line_style)
                lc_dec.set_norm(norm)
            ax.autoscale_view()
        
        if plotstyle == 'scatter' or plotstyle == 'both':
            if start_branch == 'increasing':
                c_inc = np.linspace(0, 0.5, len(xdata_inc))
                c_dec = np.linspace(0.5, 1, len(xdata_dec))
            else:
                c_inc = np.linspace(0.5, 1, len(xdata_inc))
                c_dec = np.linspace(0, 0.5, len(xdata_dec))
                
            # Plot scatter points with color gradient
            if len(xdata_inc) > 0:
                    ax.scatter(xdata_inc, ydata_inc, c=c_inc, cmap=color, norm=norm, label=label, **scatter_style)
            if len(xdata_dec) > 0:
                    ax.scatter(xdata_dec, ydata_dec, c=c_dec, cmap=color, norm=norm, **scatter_style)
        
    # plot the data points for both branches
    elif plotstyle == 'line' or plotstyle == 'both':
        if len(xdata_inc) > 0:
                ax.plot(xdata_inc, ydata_inc, label=label, color=color_inc, **line_style)
        if len(xdata_dec) > 0:
                ax.plot(xdata_dec, ydata_dec, color=color_dec, **line_style)
        if plotstyle == 'both':
            if len(xdata_inc) > 0:
                    ax.scatter(xdata_inc, ydata_inc, color=color_inc, **scatter_style)
            if len(xdata_dec) > 0:
                    ax.scatter(xdata_dec, ydata_dec, color=color_dec, **scatter_style)
    elif plotstyle == 'scatter':
        if len(xdata_inc) > 0:
                ax.scatter(xdata_inc, ydata_inc, label=label, color=color_inc, **scatter_style)
        if len(xdata_dec) > 0:
                ax.scatter(xdata_dec, ydata_dec, color=color_dec, **scatter_style)
    else:
        raise ValueError("Invalid plotstyle. Use 'line', 'scatter' or 'both' or change color to a cmap.")

    # style the plot
        ax.grid(True, **grid_style)
        # check if the data crosses the abscissa or ordinate and only plot the corresponding axis if it does
    if np.any(x < 0) and np.any(x > 0):
        ax.axvline(0, color='k', linestyle='--', zorder=1, alpha=0.9)
    if np.any(y < 0) and np.any(y > 0):
        ax.axhline(0, color='k', linestyle='--', zorder=1, alpha=0.9)
    ax.tick_params(direction='in', top=True, right=True)
    ax.set_xlabel(f'H [{xunit}]' if xunit is not None else 'H [arb. u.]')
    ax.set_ylabel(f'M [{yunit}]' if yunit is not None else 'M [arb. u.]')
    if title is not None:
        ax.set_title(title)

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            if any(lbl and not str(lbl).startswith('_') for lbl in labels):
                ax.legend(**legend_kwargs)

    return ax

# from https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html 07.04.2026
def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment
    ax.add_collection(lc)
    return lc