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
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import normalize, savgol_filter
from scipy.integrate import quad
from typing import Union
from lmfit import Model, Parameters

# import support functions
from .SupportFunctions import safe_stderr

#%%
###############################################################################        
# 2. Basic (Fit) Functions
###############################################################################
def linear(xdata: Union[float, int, list, pd.DataFrame, pd.Series, np.ndarray], a: float, b: float):
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
    xdata : float, int, list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        Input value(s) (typically named ``x`` in fit functions).
    a : float
        Slope parameter.
    b : float
        Constant offset parameter.

    Returns
    -------
    ydata : numpy.ndarray
        Modeled value(s) of the linear function.

    Raises
    ------
    ValueError
        If `xdata` is not of a supported type.

    Examples
    --------
    >>> linear([0, 1, 2], 2.0, -1.0)
    array([-1.,  1.,  3.])
    >>> linear([-3, 0, 3], 0.5, 0.0)
    array([-1.5,  0. ,  1.5])
    """
    # check xdata format
    if not isinstance(xdata, (int, float, list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'xdata must be a pandas dataframe/series, list, numpy array or int/float, not {type(xdata)}')

    xdata = np.asarray(xdata)
    return a * xdata + b

def polynomial(xdata: Union[float, int, list, pd.DataFrame, pd.Series, np.ndarray], *args: Union[float, list]):
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
    xdata : float, int, list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        Input value(s) (typically named x in functions).
    *args : float or list
        Polynomial coefficients in descending powers.
        For ``m`` coefficients, the model order is ``m-1``.
        Example:
        ``args=(2, -3, 1)`` gives ``f(x)=2x^2-3x+1``.

    Returns
    -------
    ydata : numpy.ndarray
        Modeled value(s) of the polynomial function.

    Raises
    ------
    ValueError
        If `xdata` is not of a supported type.

    Examples
    --------
    >>> polynomial([1, 2, 3], 1, 0, -1)
    array([0., 3., 8.])
    >>> polynomial([0, 1, 2], 2, -3, 1)  # 2x^2 - 3x + 1
    array([1., 0., 3.])
    """
    # check xdata format
    if not isinstance(xdata, (int, float, list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'xdata must be a pandas dataframe/series, list, numpy array or int/float, not {type(xdata)}')

    xdata = np.asarray(xdata)
    ydata = np.zeros_like(xdata, dtype=float)

    for i, coef in enumerate(args):
        ydata += coef * xdata ** (len(args) - 1 - i)

    return ydata

def arctan(
    xdata: Union[float, int, list, pd.DataFrame, pd.Series, np.ndarray], 
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
    xdata : float | int | list | numpy.ndarray | pandas.DataFrame | pandas.Series
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
    numpy.ndarray
        Modeled signal values for the provided `xdata` fields.

    Raises
    ------
    ValueError
        If `xdata` is not of a supported type.

    Examples
    --------
    >>> arctan([-10, 0, 10], a=0.0, b=1.0, c=0.1, d=0.0, e=4.0)
    array([-0.344...,  0.242...,  0.605...])
    """
    # check xdata format
    if not isinstance(xdata, (int, float, list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'xdata must be a pandas dataframe/series, list, numpy array or int/float, not {type(xdata)}')

    xdata = np.asarray(xdata)
    return a + b * 2/np.pi * np.arctan(c * (xdata - d + e))

def arctan_hys(
    xdata: Union[float, int, list, pd.DataFrame, pd.Series, np.ndarray],
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
    xdata : float, int, list, numpy.ndarray, pandas.DataFrame, or pandas.Series
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
    tuple or numpy.ndarray
        If ``xdata`` is scalar, returns ``(mean_abs, ydata1, ydata2)`` where
        ``mean_abs`` is the mean of the absolute branch values.
        If ``xdata`` is array-like, returns concatenated branch values.

    Examples
    --------
    >>> tan_hys(1.0, 0.0, 1.0, 0.2, 2.0, 5.0)
    (0.493..., 0.429..., -0.557...)
    >>> tan_hys([-6, -3, 0, 0, 3, 6], 0.0, 1.0, 0.3, 0.0, 2.0)
    array([-0.557..., -0.185..., 0.344..., -0.344..., 0.185..., 0.557...])
    """
    # if arctan of a single value is wanted. Return the mean of both branches
    # as well as the individual branches
    if isinstance(xdata, (int, float)): # for calculating single values
        ydata1 = arctan(xdata, a, b_1, c_1, d_1, e_1)
        ydata2 = arctan(xdata, a, b_1, c_1, d_1, -e_1)
        return np.mean([np.abs(ydata1), np.abs(ydata2)]), ydata1, ydata2
    
    elif isinstance(xdata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        #if xdata is given as a list (hysteresis), split it correspondingly into two branches
        xdata = np.asarray(xdata)

        # This check doesnt make sense as this is a fit function without knowledge of ydata.
        # # Check if the length of xdata is odd, i.e.the center point contributes to both branches.
        # # Duplicate the center point in this case so that both branches are equally long.
        # if len(xdata) % 2 != 0:
        #     center_index = len(xdata) // 2
        #     xdata = np.insert(xdata, center_index, xdata[center_index])

        # Split the array into two halves using slicing
        mid_index = len(xdata) // 2
        Xdata1 = xdata[:mid_index]
        Xdata2 = xdata[mid_index:]

        ydata1 = arctan(Xdata1, a, b_1, c_1, d_1, e_1)
        ydata2 = arctan(Xdata2, a, b_1, c_1, d_1, -e_1)
        return np.append(ydata1, ydata2)

def double_arctan_hys(
    xdata: Union[float, int, list, pd.DataFrame, pd.Series, np.ndarray],
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
    xdata : float, int, list, numpy.ndarray, pandas.DataFrame, or pandas.Series
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
    tuple or numpy.ndarray
        If xdata is a single value, returns a tuple of (mean, ydata1, ydata2).
        If xdata is a list or array, returns the concatenated ydata for both branches.

    Examples
    --------
    >>> double_tan_hys(1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) # for single values
    (mean_value, ydata1, ydata2)
    >>> double_tan_hys([1, 2, 3, 4, 5, 6], 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) # for lists
    array([ydata1_values, ydata2_values])
    """

    # if double arctan of a single value is wanted. Return the mean of both branches
    # as well as the individual branches
    if isinstance(xdata, (int, float)): # for calculating single values
        ydata1 =  arctan(xdata, a, b_1, c_1, d_1, e_1) + arctan(xdata, 0, b_2, c_2, d_2, e_2)
        ydata2 =  arctan(xdata, a, b_1, c_1, d_1, -e_1) + arctan(xdata, 0, b_2, c_2, d_2, -e_2)
        return np.mean([np.abs(ydata1), np.abs(ydata2)]), ydata1, ydata2
    
    elif isinstance(xdata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        # forward to tan_hys() function and its logic
        return arctan_hys(xdata, a, b_1, c_1, d_1, e_1) + arctan_hys(xdata, 0, b_2, c_2, d_2, e_2)

def mult_arctan_hys(
    xdata: Union[float, int, list, pd.DataFrame, pd.Series, np.ndarray],
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
    xdata : float, int, list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        Field value(s). Scalar input evaluates both branches at one point;
        array-like input is split into branch segments by `arctan_hys`.
    kwargs : dict
        Dictionary of model parameters.
        Required format is ``4*n + 1`` entries with keys:
        ``a, b_1, c_1, d_1, e_1, ..., b_n, c_n, d_n, e_n``.

    Returns
    -------
    tuple or numpy.ndarray
        If ``xdata`` is scalar, returns ``(mean_abs, ydata1, ydata2)`` where
        ``mean_abs`` is the mean of the absolute branch values.
        If ``xdata`` is array-like, returns concatenated branch values.

    Raises
    ------
    ValueError
        If ``kwargs`` does not follow the required naming/length convention or
        contains values of unsupported types.
    
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
        raise ValueError(f'Length of kwargs dict must be 4*n + 1, where n >= 1 is the number of arctan functions. Currently, len(kwargs) = {len(kwargs)}.')

    # Check if all key-value-pairs have the correct style:
    for key, values in kwargs.items():
        if not isinstance(values, (int, float)):
            raise ValueError(f'All values in kwargs dict must be int or float. Currently, {key} has value {values} of type {type(values)}.')
        if not isinstance(key, (str)):
            raise ValueError(f'All keys in kwargs dict must be strings. Currently, {key} is of type {type(key)}.')

        # Check if key == 'a' or key starts with 'b', 'c', 'd', or 'e' and is followed by a number
        if key != 'a' and not (key[0] in ['b', 'c', 'd', 'e'] and key.split('_')[-1].isdigit()):
            raise ValueError(f'All keys in kwargs dict must be "a" or start with "b", "c", "d", or "e" followed by a number. Currently, {key} does not follow this style.')

    n_arctan = (len(kwargs) - 1) // 4

    # Check for naming, i.e. if all n_arctan have b,c,d and e
    for n in range(1, n_arctan+1):
        keys = [f"b_{n}", f"c_{n}", f"d_{n}", f"e_{n}"]
        for key in keys:
            if key not in kwargs.keys():
                raise ValueError(f"The parameter {key} is not provided for hysteresis loop {n}.")

    if isinstance(xdata, (int, float)): # for calculating single values
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
        ydata = arctan_hys(xdata, kwargs['a'], kwargs['b_1'], kwargs['c_1'], kwargs['d_1'], kwargs['e_1'])
        if n_arctan > 1:
            for n in range(1, n_arctan):
                b = kwargs['b_' + str(n+1)]
                c = kwargs['c_' + str(n+1)]
                d = kwargs['d_' + str(n+1)]
                e = kwargs['e_' + str(n+1)]
                ydata += arctan_hys(xdata, 0, b, c, d, e)

        return ydata

#%%
###############################################################################        
# 3. Data Manipulation
###############################################################################

def invert_axis(
        xdata: Union[float, int, list, pd.DataFrame, pd.Series, np.ndarray], 
        ydata: Union[float, int, list, pd.DataFrame, pd.Series, np.ndarray], 
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
    xdata : float, int, list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        Independent variable data (e.g., applied field).
    ydata : float, int, list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        Dependent variable data (e.g., magnetization, Kerr signal).
    axis : str, optional
        Axis selection for sign inversion. Must be one of ``'x'``, ``'y'``,
        or ``'both'``. Default is ``'x'``.

    Returns
    -------
    tuple[Union[float, int, list, np.ndarray, pd.DataFrame, pd.Series], Union[float, int, list, np.ndarray, pd.DataFrame, pd.Series]]
        Sign-corrected ``(xdata, ydata)`` according to ``axis``.
        
    Raises
    ------
    ValueError
        If ``axis`` is not one of ``'x'``, ``'y'``, or ``'both'``.
        If ``xdata`` or ``ydata`` is not of a supported type.

    Examples
    --------
    >>> invert_axis(pd.DataFrame([1, 2, 3]), pd.DataFrame([4, 5, 6]), axis='x')
    (   0\n0 -1\n1 -2\n2 -3,    0\n0  4\n1  5\n2  6)
    >>> invert_axis([1, 2], [-3, -4], axis='both')
    ([-1, -2], [3, 4])
    """
    if not isinstance(xdata, (int, float, list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'xdata must be a pandas dataframe/series, list, numpy array or int/float, not {type(xdata)}')
    if not isinstance(ydata, (int, float, list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'ydata must be a pandas dataframe/series, list, numpy array or int/float, not {type(ydata)}')

    if axis == 'x':
        return -xdata, ydata
    elif axis == 'y':
        return xdata, -ydata
    elif axis == 'both':
        return -xdata, -ydata
    else:
        raise ValueError(f'Axis not recognized. Please choose between x, y, or both, not {axis}')


def del_outliers(
    ydata: Union[list, pd.DataFrame, pd.Series, np.ndarray],
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
    ydata : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        Output value(s) of the dataset (typically named y in functions).
    threshold : float, optional
        Threshold for the difference of a point to the mean of all points. The default is 2.
    neighbours : int, optional
        Number of neighbours to be taken into account for the second mean calculation as well as 
        for the linear interpolation. The default is 10.
    return_details : bool, optional
        If True, also return details about the outlier detection process. The default is False.

    Returns
    -------
    numpy.ndarray
        ydata with outliers removed and replaced by the mean of their neighbours.
    numpy.ndarray, optional
        If `return_details` is True, also returns details about the outlier detection process.

    Raises
    ------
    ValueError
        If `ydata` is not of a supported type.

    Examples
    --------
    >>> del_outliers([1, 2, 3, 100, 5, 6, 7])
    array([1, 2, 3, 4, 5, 6, 7])

    """
    # Validate input type
    if not isinstance(ydata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f"ydata must be a pandas dataframe, list or numpy array, not {type(ydata)}")

    # Convert to numpy array if it's a pandas Series or DataFrame
    ydata = np.asarray(ydata, dtype=float)
    n = len(ydata)

    # Handle empty input
    if n == 0:
        if return_details:
            details = {
                "outliers": np.array([], dtype=bool),
                "indices_outliers": np.array([], dtype=int),
            }
            return ydata, details
        return ydata

    # Local median and median absolute deviation (MAD) for outlier detection
    # MAD = median of the absolute deviations from the median, i.e. a measure for how much the data deviates.
    # The median and MAD are quite robust against single outliers
    med_neigh = np.zeros_like(ydata)
    sigma_neigh = np.zeros_like(ydata)

    for i in range(n):
        # handle edge cases by adjusting the window of neighbours
        start = max(0, i - neighbours)
        end = min(n, i + neighbours + 1)
        neigh_slice = ydata[start:end]

        # calculate median and MAD for the current window of neighbours
        med_i = np.median(neigh_slice)
        mad_i = np.median(np.abs(neigh_slice - med_i))

        # save median and MAD for the current point, convert MAD to standard deviation equivalent using the constant 1.4826 for normal distribution
        # See https://en.wikipedia.org/wiki/Median_absolute_deviation
        med_neigh[i] = med_i
        sigma_neigh[i] = 1.4826 * mad_i if mad_i > 0 else 1e-6  # Avoid division by zero

    # Calculate local outlier score (deviation from local median, relative to local MAD)
    local_score = np.abs(ydata - med_neigh) / sigma_neigh

    # Check if local score exceeds the threshold to identify outliers
    outliers = local_score > threshold
    outlier_indices = np.where(outliers)[0]

    # Replace outliers with local interpolation (mean of neighbours excluding the outlier)
    for outlier in outlier_indices:
        if outlier:
            start = max(0, outlier - neighbours)
            end = min(outlier + neighbours + 1, n)
            neigh_slice = np.concatenate([ydata[start:outlier], ydata[outlier + 1:end]])
            if len(neigh_slice):
                ydata[outlier] = np.mean(neigh_slice)

    if return_details:
        details = {
            "outliers": outliers,
            "indices_outliers": outlier_indices,
            "local_score": local_score,
        }
        return ydata, details

    return ydata

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
    pandas.Series
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
    ydata: Union[list, pd.DataFrame, pd.Series, np.ndarray], 
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
    ydata : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        Output value(s) of the dataset (typically named y in functions).
    sat_region : float, optional
        Amount of points at the beginning and end of the hysteresis to be taken into account for the mean calculation.
        The default is 0.05, which represents the first and last 5% of points.
        This is equal to sat_regions=0.1 in the other functions for symmetric loops, where the saturation is calculated based on the applied field.
    return_details : bool, optional
        If True, returns additional details about the correction process. The default is False.

    Returns
    -------
    np.ndarray
        Array of magnetization M or intensity with an opening below the noise level at the end of the hysteresis. 
        Does not change the opening if no significant opening (relative to the noise) is present.

    Raises
    ------
    ValueError
        If `sat_region` is greater than or equal to 0.5.
        If `ydata` is not of a supported type.

    Examples
    --------
    >>> rmv_opening([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.1)
    array([1. , 1.9, 2.8, 3.7, 4.6, 5.5, 6.4, 7.3, 8.2, 9.1])
    """
    if sat_region >= 0.5:
        raise ValueError('Amount must be smaller than 0.5 (half of the hysteresis)')
    
    if not isinstance(ydata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'ydata must be a pandas dataframe, list or numpy array, not {type(ydata)}')
    
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
    
    # subtract slope from ydata. Reminder slope is in time/point number and not in xdata/field strength
    ydata -= opening_slope * np.arange(len(ydata))
    
    if return_details:
        return ydata, {
            'opening_slope': opening_slope, 
            'mean_diff': mean_diff,
            'standard_deviation': standard_deviation,
            'sat_points': sat_points,
            }
    return ydata

def slope_correction(
    xdata: Union[list, pd.DataFrame, pd.Series, np.ndarray],
    ydata: Union[list, pd.DataFrame, pd.Series, np.ndarray],
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
    xdata : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        Input value(s) of the dataset (typically named x in functions).
    ydata : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
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
    np.ndarray
        Array of magnetic moment M or intensity without a constant slope, i.e., flat at the border regions if no higher-order effects are present.

    Raises
    ------
    ValueError
        If `xdata` or `ydata` is not of a supported type.
        If no saturation region is found.

    Examples
    --------
    >>> slope_correction([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    array([1.9, 2.9, 3.9, 4.9, 5.9])
    """
    if not isinstance(xdata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'xdata must be a pandas dataframe, list or numpy array, not {type(xdata)}')
    if not isinstance(ydata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'ydata must be a pandas dataframe, list or numpy array, not {type(ydata)}')
    
    # convert to numpy arrays if not already
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # define sat_region in field strength
    sat_field_range = sat_region * (np.max(xdata) - np.min(xdata))
    # take end regions of hysteresis (saturated regions) by calculating the field
    # strengths which are assumed to be in saturation
    upper_saturation_limit = np.max(xdata) - sat_field_range
    lower_saturation_limit = np.min(xdata) + sat_field_range

    # take end regions of hysteresis (saturated regions of both branches)
    upper_saturation_region = xdata > upper_saturation_limit
    lower_saturation_region = xdata < lower_saturation_limit

    if not np.any(upper_saturation_region) or not np.any(lower_saturation_region):
        raise ValueError('No saturation region found')
    
    # fit linear functions to both ends
    popt1, pcov1 = curve_fit(linear, xdata[upper_saturation_region], ydata[upper_saturation_region])
    popt2, pcov2 = curve_fit(linear, xdata[lower_saturation_region], ydata[lower_saturation_region])
    # calculated mean slope of both hysteresis ends
    slope = np.mean([popt1[0], popt2[0]])

    # Calculate noise level (as standard deviation of the residuals)
    residuals_upper = ydata[upper_saturation_region] - linear(xdata[upper_saturation_region], *popt1)
    residuals_lower = ydata[lower_saturation_region] - linear(xdata[lower_saturation_region], *popt2)
    noise_level = noise_threshold * np.mean([np.std(residuals_upper), np.std(residuals_lower)])

    # Check if the slope (effect over the field range) is insignificant (below the noise level) and do nothing if it is
    slope_effect = np.abs(slope) * (np.max(xdata) - np.min(xdata))
    norm_slope_diff = np.abs(1 - popt1[0]/popt2[0]) if popt2[0] != 0 else np.inf
    if slope_effect < noise_level or norm_slope_diff > branch_difference:
        if return_details:
            return ydata, {
                'slope': slope, 
                'noise_level': noise_level,
                'slope_effect': slope_effect,
                'norm_slope_diff': norm_slope_diff
                }
        return ydata

    ydata_corrected = ydata - slope * xdata
    # otherwise return subtracted/corrected magnetization
    if return_details:
            return ydata_corrected, {
                'slope': slope, 
                'noise_level': noise_level,
                'slope_effect': slope_effect,
                'norm_slope_diff': norm_slope_diff
                }

def hys_norm(
    xdata: Union[list, pd.DataFrame, pd.Series, np.ndarray],
    ydata: Union[list, pd.DataFrame, pd.Series, np.ndarray],
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
    xdata : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        List of externally applied field strengths H.
    ydata : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        List of magnetization values M.
    sat_region : float, optional
        Outermost fraction of the xdata which is assumed to be in saturation.
        Default is 0.1, i.e., 10% of the outermost xdata is assumed to be in saturation.
    return_details : bool, optional
        If True, returns additional details about the normalization process. The default is False.

    Returns
    -------
    np.ndarray
        Normalized list of magnetization in the range of roughly -1 to +1.

    Raises
    ------
    ValueError
        If `xdata` or `ydata` is not of a supported type.
        If no saturation region is found.

    Examples
    --------
    >>> hys_norm([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    array([-1. , -0.5,  0. ,  0.5,  1. ])
    """
    if not isinstance(xdata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'xdata must be a pandas dataframe, list or numpy array, not {type(xdata)}')
    if not isinstance(ydata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'ydata must be a pandas dataframe, list or numpy array, not {type(ydata)}')

    # convert to numpy arrays if not already
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    
    # take end regions of hysteresis (saturated regions)
    upper_saturation_limit = (1 - sat_region) * np.max(xdata)
    lower_saturation_limit = (1 - sat_region) * np.min(xdata)

    # take end regions of hysteresis (saturated regions)
    upper_saturation_region = xdata > upper_saturation_limit
    lower_saturation_region = xdata < lower_saturation_limit

    if not np.any(upper_saturation_region) or not np.any(lower_saturation_region):
        raise ValueError('No saturation region found')
    else:
        # average saturated regions
        lmax = np.mean(ydata[upper_saturation_region])
        lmin = np.mean(ydata[lower_saturation_region])
        # calculate shift/bias and normalization
        y_bias = 0.5 * (lmax + lmin)
        norm = 0.5 * (lmax - lmin)
        # return normalized magnetization
        if return_details:
            return (ydata - y_bias) / norm, {'y_bias': y_bias, 'norm': norm}

        return (ydata - y_bias) / norm

def hys_center(
    xdata: Union[list, pd.DataFrame, pd.Series, np.ndarray],
    ydata: Union[list, pd.DataFrame, pd.Series, np.ndarray],
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
    xdata : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        List of externally applied field strengths H.
    ydata : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
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
        - xdata : numpy.ndarray
            Adjusted list of externally applied field strengths H.
        - ydata : numpy.ndarray
            Centered (and optionally normalized) list of magnetization values M.

    Raises
    ------
    ValueError
        If `xdata` and `ydata` do not have the same length.
        If no saturation region is found.
        If `xdata` or `ydata` is not of a supported type.

    Examples
    --------
    >>> hys_center([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    (array([1, 2, 3, 4, 5]), array([-1. , -0.5,  0. ,  0.5,  1. ]))
    """
    if not isinstance(xdata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'xdata must be a pandas dataframe, list or numpy array, not {type(xdata)}')
    if not isinstance(ydata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'ydata must be a pandas dataframe, list or numpy array, not {type(ydata)}')
    
    # convert to numpy arrays if not already
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    
    # xdata and ydata must have the same length and this length must be even
    if len(xdata) != len(ydata):
        raise ValueError('xdata and ydata must have the same length')
    # Check if the length of xdata is odd, i.e., the center point contributes to both branches.
    # Duplicate the center point in this case so that both branches are equally long.
    if len(xdata) % 2 != 0:
        center_index = len(xdata) // 2
        xdata = np.insert(xdata, center_index, xdata[center_index])
        ydata = np.insert(ydata, center_index, ydata[center_index])

    # Center in ydata: center between saturations

    # take end regions of hysteresis (saturated regions)
    upper_saturation_limit = (1 - sat_region) * np.max(xdata)
    lower_saturation_limit = (1 - sat_region) * np.min(xdata)
    upper_saturation_region = xdata > upper_saturation_limit
    lower_saturation_region = xdata < lower_saturation_limit

    if not np.any(upper_saturation_region) or not np.any(lower_saturation_region):
        raise ValueError('No saturation region found')
    else:
        # average saturated regions
        lmax = np.mean(ydata[upper_saturation_region])
        lmin = np.mean(ydata[lower_saturation_region])
        # calculate shift/bias and normalization
        y_bias = 0.5 * (lmax + lmin)
        norm = 0.5 * (lmax - lmin)
        if normalize:
            # return normalized magnetization
            ydata = (ydata - y_bias) / norm
        else:
            # return centered magnetization
            ydata = ydata - y_bias

    # Center in xdata: center branches if difference is above one step size
    
    # Calculate the step size of the xdata
    xdata_step = np.mean(np.abs(np.diff(xdata)))
    # Calculate the difference between the branches
    diff = np.mean(ydata[:len(ydata)//2]) - np.mean(ydata[len(ydata)//2:]) # if < 0, branch 1 is higher than branch 2

    # Check if the difference between the branches is larger than one step size
    if np.abs(diff) > xdata_step:
        # Calculate the shift in xdata
        x_shift = diff / 2
        # Shift xdata
        xdata[:len(xdata)//2] -= x_shift
        xdata[len(xdata)//2:] += x_shift

    if return_details:
        return xdata, ydata, {'x_shift': x_shift, 'y_bias': y_bias, 'norm': norm}
    else:
        return xdata, ydata

#%%
###############################################################################        
# 4. Data Evaluation
###############################################################################

def x_sect(xdata: pd.Series, ydata: pd.Series, offset: float = 0, steepness_for_fit: bool = False):
    """
    Calculate the first intersection of a hysteresis loop with the x-axis.

    This function takes a hysteresis loop with xdata (external field H) and ydata (magnetization M), 
    typically of a single branch, and calculates the first intersection with the x-axis using linear 
    interpolation between two subsequent points with changing sign of their y-values.

    Note: This function may be non-robust to strong noise. Consider using functions that check for a 
    clear change in sign by comparing the closest neighbours.

    Parameters
    ----------
    xdata : pd.Series or np.ndarray
        List of externally applied field strengths H, typically of a single branch.
    ydata : pd.Series or np.ndarray
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
    ValueError
        If `xdata` and `ydata` do not have the same length.
        If `xdata` or `ydata` is not of a supported type.

    Examples
    --------
    >>> x_sect(pd.Series([1, 2, 3]), pd.Series([-1, 0, 1]))
    (2.0, 0.0)
    >>> x_sect(pd.Series([1, 2, 3]), pd.Series([-1, 0, 1]), offset=0.5, steepness_for_fit=True)
    (2.0, 0.0, 1.0)
    """
    
    if len(xdata) != len(ydata):
        raise ValueError('xdata and ydata must have the same length')
    
    if not isinstance(xdata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'xdata must be a pandas dataframe, list or numpy array, not {type(xdata)}')
    
    if not isinstance(ydata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'ydata must be a pandas dataframe, list or numpy array, not {type(ydata)}')
    
    if np.abs(offset) > np.max(np.abs(ydata)):
        raise ValueError('Offset is larger than the maximum absolute value of ydata, no intersection with the offset can be found')
    
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata).copy() - offset # shift ydata by offset to find intersection with the offset instead of the x-axis
    
    # Ensure that the ydata list starts with a negative value (negative saturation)
    if list(ydata)[0] > 0.0: 
        ydata = np.flipud(ydata)
        xdata = np.flipud(xdata)

    # Initialize variables
    intersect = 0.0
    intersect_err = 0.0
    a = 0.0

    # Check for points where the product of two adjacent points is negative or equal to zero
    for i in range(1, len(ydata)):
        product = ydata[i-1] * ydata[i]
        if product <= 0:
            # If the product is zero, the intersection is directly found
            if ydata[i] == 0:
                return xdata[i], 0.0
            else:
                # Linearly interpolate between the two points
                a = (ydata[i] - ydata[i-1]) / (xdata[i] - xdata[i-1])
                if a == np.inf or a == -np.inf: # Rarely, xdata[i] and xdata[i-1] are identical leading to a division by zero. Then the slope is wrongly calculated as inf or -inf. Happend once in 2 years of usage.
                    a = (ydata[i] - ydata[i-2]) / (xdata[i] - xdata[i-2])
                if a != 0:
                    b = ydata[i-1] - a * xdata[i-1]
                    intersect = -b / a
                    intersect_err = max(np.abs(intersect - xdata[i]), np.abs(intersect - xdata[i-1]))

    if steepness_for_fit:
        return intersect, intersect_err, a
    else:
        return intersect, intersect_err
    
def y_sect(xdata: pd.Series, ydata: pd.Series, offset: float = 0):
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
    xdata : pd.Series or np.ndarray
        List of externally applied field strengths H, typically of a single branch.
    ydata : pd.Series or np.ndarray
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
    ValueError
        If `xdata` and `ydata` do not have the same length.
        If `xdata` or `ydata` is not of a supported type.

    Examples
    --------
    >>> y_sect(pd.Series([1, 2, 3]), pd.Series([-1, 0, 1]))
    (0.0, 0.0)
    >>> y_sect(pd.Series([1, 2, 3]), pd.Series([-1, 0, 1]), offset=1)
    (0.0, 0.0)
    """
    
    if len(xdata) != len(ydata):
        raise ValueError('xdata and ydata must have the same length')
    
    if not isinstance(xdata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'xdata must be a pandas dataframe, list or numpy array, not {type(xdata)}')
    
    if not isinstance(ydata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'ydata must be a pandas dataframe, list or numpy array, not {type(ydata)}')
    
    if np.abs(offset) > np.max(np.abs(xdata)):
        raise ValueError('Offset is larger than the maximum absolute value of xdata, no intersection with the offset can be found')
    
    xdata = np.asarray(xdata).copy() - offset # shift xdata by offset to find intersection with the offset instead of the y-axis
    ydata = np.asarray(ydata)
    
    # Ensure that the xdata list starts with a negative value (from left to right)
    if list(xdata)[0] > 0.0:
        ydata = np.flipud(ydata)
        xdata = np.flipud(xdata)

    # Initialize variables
    intersect = 0.0
    intersect_err = 0.0

    #  Check for points where the product of two adjacent points in x is negative or equal to zero
    for i in range(1, len(xdata)):
        product = xdata[i-1] * xdata[i]
        if product <= 0:
            # If the product is zero, the intersection is directly found
            if xdata[i] == 0:
                return ydata[i], 0.0
            else:
                # Linearly interpolate between the two points
                a = (ydata[i] - ydata[i-1]) / (xdata[i] - xdata[i-1])
                if a == np.inf or a == -np.inf: # Rarely, xdata[i] and xdata[i-1] are identical leading to a division by zero. Then the slope is wrongly calculated as inf or -inf. Happend once in 2 years of usage.
                    a = (ydata[i] - ydata[i-2]) / (xdata[i] - xdata[i-2])
                if a != 0:
                    b = ydata[i-1] - a * xdata[i-1]
                    intersect = b
                    intersect_err = max(np.abs(intersect - ydata[i]), np.abs(intersect - ydata[i-1]))
                    
    return intersect, intersect_err

def num_derivative(xdata: pd.Series, ydata: pd.Series):
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
    ValueError
        If `xdata` and `ydata` do not have the same length.
        If `xdata` or `ydata` is not of a supported type.

    Examples
    --------
    >>> num_derivative(pd.Series([3, 5, 7]), pd.Series([1, 2, 3]))
    (array([4., 6.]), array([0.5, 0.5]))
    """
    
    if len(xdata) != len(ydata):
        raise ValueError('xdata and ydata must have the same length')
    
    if not isinstance(xdata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'xdata must be a pandas dataframe, list or numpy array, not {type(xdata)}')
    
    if not isinstance(ydata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'ydata must be a pandas dataframe, list or numpy array, not {type(ydata)}')
    
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    
    # Calculate the derivative
    der_ydata = np.diff(ydata) / np.diff(xdata)
    # Calculate the xdata in between the input xdata
    der_xdata = xdata[:-1] + np.diff(xdata) / 2
    
    return der_xdata, der_ydata

def num_integral(xdata: pd.Series, ydata: pd.Series):
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
    ValueError
        If `xdata` and `ydata` do not have the same length.
        If `xdata` or `ydata` is not of a supported type.

    Examples
    --------
    >>> num_integral(pd.Series([3, 5, 7]), pd.Series([1, 2, 3]))
    (array([4., 6.]), array([2., 2.]))
    """
    if len(xdata) != len(ydata):
        raise ValueError('xdata and ydata must have the same length')
    
    if not isinstance(xdata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'xdata must be a pandas dataframe, list or numpy array, not {type(xdata)}')
    
    if not isinstance(ydata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'ydata must be a pandas dataframe, list or numpy array, not {type(ydata)}')
    
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    
    # Calculate the integral
    int_ydata = (ydata[:-1] + ydata[1:]) / 2 * np.abs(np.diff(xdata))
    # Calculate the xdata in between the input xdata
    int_xdata = xdata[:-1] + np.diff(xdata) / 2
    
    return int_xdata, int_ydata

def lin_hyseval(
        xdata, 
        ydata, 
        sat_region: float = 0.95,
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
    xdata : list or np.ndarray
        List of externally applied field strengths H, typically of a single branch.
    ydata : list or np.ndarray
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
    ValueError
        If `xdata` and `ydata` do not have the same length.
        If `xdata` or `ydata` is not of a supported type.

    Examples
    --------
    >>> lin_hyseval([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    {'HEB': 2.5, 'dHEB': 0.1, 'HC': 1.0, 'dHC': 0.1, 'MR': (0.5, 1.0, 0.0), 'dMR': 0.1, 'MHEB': 0.5, 'dMHEB': 0.1}
    >>> lin_hyseval([1, 2, 3, 4, 5], [2, 3, 4, 5, 6], steepness_for_fit=True)
    {'HEB': 2.5, 'dHEB': 0.1, 'HC': 1.0, 'dHC': 0.1, 'MR': (0.5, 1.0, 0.0), 'dMR': 0.1, 'MHEB': 0.5, 'dMHEB': 0.1, 'a1': 1.0, 'a2': 1.0}
    """
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # Check if the length of xdata is odd, i.e., the center point contributes to both branches.
    # Duplicate the center point in this case so that both branches are equally long.
    if len(xdata) % 2 != 0:
        center_index = len(xdata) // 2
        xdata = np.insert(xdata, center_index, xdata[center_index])
        ydata = np.insert(ydata, center_index, ydata[center_index])
    
    if use_offset:
        # take end regions of hysteresis (saturated regions)
        upper_saturation_limit = (1 - sat_region) * np.max(xdata)
        lower_saturation_limit = (1 - sat_region) * np.min(xdata)
        upper_saturation_region = xdata > upper_saturation_limit
        lower_saturation_region = xdata < lower_saturation_limit

        if not np.any(upper_saturation_region) or not np.any(lower_saturation_region):
            raise ValueError('No saturation region found')
            magoffset = 0
        
        else:
            # average saturated regions
            magmax = np.mean(ydata[upper_saturation_region])
            magmin = np.mean(ydata[lower_saturation_region])
            # calculate shift/bias and normalization
            magoffset = 0.5 * (magmax + magmin)
            norm = 0.5 * (magmax - magmin)
    
    else:
        magoffset = 0

    # Obtain intersections as coercive fields with the x_sect function
    # Split the array into two halves using slicing
    mid_index = len(xdata) // 2
    Xdata1 = xdata[:mid_index]
    Xdata2 = xdata[mid_index:]

    Ydata1 = ydata[:mid_index]
    Ydata2 = ydata[mid_index:]
    # branch-dependently
    HC1, dHC1, a1 = x_sect(Xdata1, Ydata1, offset=magoffset, steepness_for_fit=True) # first branch
    HC2, dHC2, a2 = x_sect(Xdata2, Ydata2, offset=magoffset, steepness_for_fit=True) # second branch

    half_step_size = np.mean(np.abs(np.diff(xdata))) / 2
    
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
    dMR = (dMR1 + dMR2) / 2 # uncertainty via propagation of uncertainty

    # Magnetization at the exchange bias field
    MHEB1, dMHEB1 = y_sect(Xdata1, Ydata1, offset=HEB)
    MHEB2, dMHEB2 = y_sect(Xdata2, Ydata2, offset=HEB)
    # Average of both branches
    MHEB = ((np.abs(MHEB1) + np.abs(MHEB2)) / 2, MHEB1, MHEB2)
    dMHEB = (dMHEB1 + dMHEB2) / 2 # uncertainty via propagation of uncertainty

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
    xdata: Union[list, np.ndarray, pd.Series], 
    ydata: Union[list, np.ndarray, pd.Series],
    sat_cond: float = 0.95,
    sat_region: float = 0.95,
    use_offset: bool = True,
    param_estimates: dict = None,
    param_bounds: dict = None,
    param_fixed: dict = None,
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
    xdata : list or np.ndarray
        List of externally applied field strengths H.
    ydata : list or np.ndarray
        List of magnetization values M.
    sat_cond : float, optional
        Saturation condition for the tanh function. Default is 0.95.
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
            - x_unit : str or None
                Unit of xdata.
            - y_unit : str or None
                Unit of ydata.

    Raises
    ------
    ValueError
        If `xdata` and `ydata` do not have the same length.
        If `xdata` or `ydata` is not of a supported type.

    Examples
    --------
    >>> tan_hyseval([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    ({'xdata': ..., 'ydata': ..., 'xdata_err': ..., 'ydata_err': ...},
     {'r_squared': ..., 'HEB': ..., 'dHEB': ..., 'HC': ..., 'dHC': ..., 'MR': ..., 'dMR': ..., 'MHEB': ..., 'dMHEB': ..., 'integral': ..., 'dintegral': ..., 'saturation_fields': ..., 'dsaturation_fields': ..., 'slope_atHC': ..., 'dslope_atHC': ..., 'slope_atHEB': ..., 'dslope_atHEB': ..., 'alpha': ..., 'dalpha': ..., 'rectangularity': ..., 'drectangularity': ..., 'x_unit': ..., 'y_unit': ...})
    """
    
    # Check if the length of xdata is odd, i.e.the center point contributes to both branches.
    # Duplicate the center point in this case so that both branches are equally long.
    if len(xdata) % 2 != 0:
        center_index = len(xdata) // 2
        xdata = np.insert(xdata, center_index, xdata[center_index])
        ydata = np.insert(ydata, center_index, ydata[center_index])
    
    # Check if xdata and ydata have the same length
    if len(xdata) != len(ydata):
        raise ValueError('xdata and ydata must have the same length')
    
    # Check if xdata and ydata are pd.Series and if they have units, extract the units for later use
    if isinstance(xdata, pd.Series):
        x_unit = xdata.attrs.get("unit") if hasattr(xdata, 'unit') else None
        xdata = xdata.to_numpy()
    if isinstance(ydata, pd.Series):
        y_unit = ydata.attrs.get("unit") if hasattr(ydata, 'unit') else None
        ydata = ydata.to_numpy()
    
    # quick linear calculation to determine initial guesses for the exchange bias and the coercive field strength
    LIN = lin_hyseval(xdata, ydata, sat_region=sat_region, use_offset=use_offset, steepness_for_fit=True)
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
    params.add('a', value=(np.max(ydata) + np.min(ydata))/2) # offset
    params.add('b_1', value=(np.max(ydata) - np.min(ydata))/2) # amplitude
    params.add('c_1', value=slope / params['b_1'].value, min=0) # steepness
    params.add('d_1', value=HEB_tmp, min=np.min(xdata), max=np.max(xdata)) # exchange bias field
    params.add('e_1', value=HC_tmp, min=0, max=(np.max(xdata) - np.min(xdata)) / 2) # coercive field

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
    result = model.fit(ydata, params, calc_covar=True, method=method, xdata=xdata,)

    # Calculate fitted magnetization values
    ydata_fitted = result.best_fit
    #convert to pd.Series for consistency
    ydata_fitted = pd.Series(ydata_fitted)
    if y_unit:
        ydata_fitted.unit = y_unit

    # calculate the 3 sigma uncertainty of the fit
    ydata_fitted_err = result.eval_uncertainty(sigma=3)
    # convert to pd.Series for consistency
    ydata_fitted_err = pd.Series(ydata_fitted_err)
    if y_unit:
        ydata_fitted_err.unit = y_unit

    # create a copy of the averaged xdata (branch 1 and 2) with which the results are
    # displayed. Only important for plotting the area of the hysteresis loop. As long as
    # the xdata is symmetric, the average is symmetric as well.
    # Split the array into two halves using slicing
    mid_index = len(xdata) // 2
    Xdata1 = xdata[:mid_index]
    Xdata2 = xdata[mid_index:]
    xdata_fitted = np.mean([Xdata2[::-1], Xdata1], axis=0)
    xdata_fitted = pd.Series([*xdata_fitted, *xdata_fitted[::-1]])
    if x_unit:
        xdata_fitted.unit = x_unit

    #xdata uncertainty = half step size
    #TODO: For Kerr and VSM the step size may change, so this should be calculated from the data
    #TODO: np.diff without mean? In that case: HC/HEB error should be calculated at their respective positions
    #TODO: And I have to make sure that np.diff has the same length as xdata
    half_step_size = np.mean(np.abs(np.diff(xdata_fitted))) / 2
    #convert to pd.Series for consistency
    xdata_fitted_err = pd.Series([half_step_size] * len(xdata_fitted))
    if x_unit:
        xdata_fitted_err.unit = x_unit

    fitted_data = {
        'xdata': xdata_fitted, 
        'ydata': ydata_fitted,
        'xdata_err': xdata_fitted_err,
        'ydata_err': ydata_fitted_err,
    }

    # params = {
    #     'r_squared': result.rsquared,
    #     'HEB': HEB, 
    #     'dHEB': dHEB,
    #     'HC': HC, 
    #     'dHC': dHC,
    #     'MS': MS,
    #     'dMS': dMS,
    #     'MR': MR,
    #     'dMR': dMR,
    #     'MHEB': MHEB,
    #     'dMHEB': dMHEB,
    #     'integral': integral,
    #     'dintegral': integral_err,
    #     'saturation_fields': saturation_fields,
    #     'dsaturation_fields': dsaturation_fields,

    #     'slope_atHC': slope_atHC,
    #     'dslope_atHC': dslope_atHC,
    #     'slope_atHEB': slope_atHEB,
    #     'dslope_atHEB': dslope_atHEB,
    #     'alpha': alpha,
    #     'dalpha': dalpha,
    #     'rectangularity': rectangularity,
    #     'drectangularity': drectangularity,

    #     'x_unit': None,
    #     'y_unit': None,

    #     # fit parameters
    #     'a': result.params['a'].value,
    #     'da': safe_stderr(result.params['a'].stderr),
    #     'b': result.params['b'].value,
    #     'db': safe_stderr(result.params['b'].stderr),
    #     'c': result.params['c'].value,
    #     'dc': safe_stderr(result.params['c'].stderr),
    #     'd': result.params['d'].value,
    #     'dd': safe_stderr(result.params['d'].stderr),
    #     'e': result.params['e'].value,
    #     'de': safe_stderr(result.params['e'].stderr),

    #     }
    
    params = arctan_hyseval_params(result, xdata, ydata, sat_cond)
    
    return fitted_data, params, result

def double_arctan_hyseval(
    xdata: Union[list, np.ndarray, pd.Series], 
    ydata: Union[list, np.ndarray, pd.Series], 
    sat_cond: float = 0.95,
    sat_region: float = 0.95,
    use_offset: bool = True,
    param_estimates: dict = None,
    param_bounds: dict = None,
    param_fixed: dict = None,
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
    xdata : list or np.ndarray
        List of externally applied field strengths H.
    ydata : list or np.ndarray
        List of magnetization values M.
    sat_cond : float, optional
        Saturation condition for the tanh function. Default is 0.95.
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
            # - ratio : dict #TODO: Implement ratio calculation
            #     Dictionary containing:
            #     - HEB1/HEB2 : float
            #         Ratio of exchange bias fields.
            #     - HC1/HC2 : float
            #         Ratio of coercive fields.
            #     - A1/A2 : float
            #         Ratio of amplitudes.
            #     - area1/area2 : float
            #         Ratio of areas.
            - x_unit : str or None
                Unit of xdata.
            - y_unit : str or None
                Unit of ydata.

    Raises
    ------
    ValueError
        If `xdata` and `ydata` do not have the same length.
        If `xdata` or `ydata` is not of a supported type.
    """
    # Check if the length of xdata is odd, i.e.the center point contributes to both branches.
    # Duplicate the center point in this case so that both branches are equally long.
    if len(xdata) % 2 != 0:
        center_index = len(xdata) // 2
        xdata = np.insert(xdata, center_index, xdata[center_index])
        ydata = np.insert(ydata, center_index, ydata[center_index])
            
    if hasattr(xdata, 'unit'):
        x_unit = xdata.unit
    else:
        x_unit = None
    if hasattr(ydata, 'unit'):
        y_unit = ydata.unit
    else:
        y_unit = None
    # quick linear calculation to determine initial guesses for the exchange bias and the coercive field strength
    LIN = lin_hyseval(xdata, ydata, sat_region=sat_region, use_offset=use_offset)
    HEB_tmp, HC_tmp = LIN['HEB'], LIN['HC']
    
    # Create a model from the function
    model = Model(double_arctan_hys)

    # Define the parameters
    params = Parameters()
    params.add('a', value=0.0) # offset, 0 for normalized hysteresis with pos/neg Sat.
    params.add('b_1', value=(np.max(ydata) - np.min(ydata))/4) # amplitude, a quarter of the spread of the hysteresis, assuming that the two loops are of similar size
    params.add('c_1', value=5.0) # steepness
    params.add('d_1', value=HEB_tmp - 0.2 * np.abs(np.min(xdata)), min=np.min(xdata), max=np.max(xdata)) # lower exchange bias field
    params.add('e_1', value=HC_tmp, min=0, max=np.max(xdata) - np.min(xdata)) # coercive field
    params.add('b_2', value=(np.max(ydata) - np.min(ydata))/4) # amplitude, a quarter of the spread of the hysteresis
    params.add('c_2', value=5.0) # steepness
    # d_2 has to be larger or equal to d
    params.add('d_2', value=HEB_tmp + 0.2 * np.abs(np.max(xdata)), min=np.min(xdata), max=np.max(xdata)) # upper exchange bias field
    params.add('e_2', value=HC_tmp, min=0, max=np.max(xdata) - np.min(xdata)) # coercive field

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
    result = model.fit(ydata, params, calc_covar=True, method=method, xdata=xdata,)

    # Calculate fitted magnetization values
    ydata_fitted = result.best_fit
    #convert to pd.Series for consistency
    ydata_fitted = pd.Series(ydata_fitted)
    if y_unit:
        ydata_fitted.unit = y_unit

    # calculate the 3 sigma uncertainty of the fit
    ydata_fitted_err = result.eval_uncertainty(sigma=3)
    # convert to pd.Series for consistency
    ydata_fitted_err = pd.Series(ydata_fitted_err)
    if y_unit:
        ydata_fitted_err.unit = y_unit

    # create a copy of the averaged xdata (branch 1 and 2) with which the results are
    # displayed
    # Split the array into two halves using slicing
    mid_index = len(xdata) // 2
    Xdata1 = xdata[:mid_index]
    Xdata2 = xdata[mid_index:]
    xdata_fitted = np.mean([Xdata2[::-1], Xdata1], axis=0)
    xdata_fitted = pd.Series([*xdata_fitted, *xdata_fitted[::-1]])
    if x_unit:
        xdata_fitted.unit = x_unit

    #xdata uncertainty = half step size
    half_step_size = np.mean(np.abs(np.diff(xdata_fitted))) / 2
    #convert to pd.Series for consistency
    xdata_fitted_err = pd.Series([half_step_size] * len(xdata_fitted))
    if x_unit:
        xdata_fitted_err.unit = x_unit

    fitted_data = {
        'xdata': xdata_fitted, 
        'ydata': ydata_fitted,
        'xdata_err': xdata_fitted_err,
        'ydata_err': ydata_fitted_err,
    }
    
    # tan_hyseval_params orders the fitted values by 1. increasing HEB and 2. by decreasing HC
    # So for the decreasing branch, the field numbers correspond to the switching behavior from 
    # left to the right. Unless for the unlikely but possible event a very large coercive field 
    # strength with a more higher EB field for the second loop
    params = arctan_hyseval_params(result, xdata, ydata, sat_cond=sat_cond)
    
    return fitted_data, params, result

def mult_arctan_hyseval(
    xdata: Union[list, np.ndarray, pd.Series], 
    ydata: Union[list, np.ndarray, pd.Series], 
    sat_cond: float = 0.95,
    sat_region: float = 0.95,
    use_offset: bool = True,
    n_arctan: int = 3, # for n = 1 or 2 the previous functions make more sense
    arctan_types: list = None,
    param_estimates: dict = None,
    param_bounds: dict = None,
    param_fixed: dict = None,
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
        Saturation condition for the tanh function. Default is 0.95.
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
    # Check if the length of xdata is odd, i.e.the center point contributes to both branches.
    # Duplicate the center point in this case so that both branches are equally long.
    if len(xdata) % 2 != 0:
        center_index = len(xdata) // 2
        xdata = np.insert(xdata, center_index, xdata[center_index])
        ydata = np.insert(ydata, center_index, ydata[center_index])
            
    if hasattr(xdata, 'unit'):
        x_unit = xdata.unit
    else:
        x_unit = None
    if hasattr(ydata, 'unit'):
        y_unit = ydata.unit
    else:
        y_unit = None
    # quick linear calculation to determine initial guesses for the exchange bias and the coercive field strength
    LIN = lin_hyseval(xdata, ydata, sat_region=sat_region, use_offset=use_offset)
    HEB_tmp, HC_tmp = LIN['HEB'], LIN['HC']

    # Define the parameters
    params = Parameters()
    params.add('a', value=np.mean(ydata)) # offset, assume mean of the data for better convergence
    if arctan_types is None:
        for n in range(1, n_arctan + 1):
            params.add(f'b_{n}', value=(np.max(ydata) - np.min(ydata))/(2*n_arctan)) # amplitude, equal portion of the hyst's mag
            params.add(f'c_{n}', value=5.0, min=0) # steepness
            params.add(f'd_{n}', value=HEB_tmp + 0.02 * np.abs(np.min(xdata)) * n, min=np.min(xdata), max=np.max(xdata)) # lower exchange bias field
            params.add(f'e_{n}', value=HC_tmp, min=0, max=np.max(xdata) - np.min(xdata)) # coercive field

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

            params.add(f'b_{n}', value=(np.max(ydata) - np.min(ydata))/(2*n_arctan)) # amplitude, equal portion of the hyst's mag
            params.add(f'c_{n}', value=5.0) # steepness
            params.add(f'd_{n}', value=HEB_tmp + 0.02 * np.abs(np.min(xdata)) * n, min=np.min(xdata), max=np.max(xdata)) # lower exchange bias field
            params.add(f'e_{n}', value=HC_tmp, min=0, max=np.max(xdata) - np.min(xdata)) # coercive field

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
    result = model.fit(ydata, params, calc_covar=True, method=method, xdata=xdata,)

    # Calculate fitted magnetization values
    ydata_fitted = result.best_fit
    #convert to pd.Series for consistency
    ydata_fitted = pd.Series(ydata_fitted)
    if y_unit:
        ydata_fitted.unit = y_unit

    # calculate the 3 sigma uncertainty of the fit
    ydata_fitted_err = result.eval_uncertainty(sigma=3)
    # convert to pd.Series for consistency
    ydata_fitted_err = pd.Series(ydata_fitted_err)
    if y_unit:
        ydata_fitted_err.unit = y_unit

    # create a copy of the averaged xdata (branch 1 and 2) with which the results are
    # displayed
    # Split the array into two halves using slicing
    mid_index = len(xdata) // 2
    Xdata1 = xdata[:mid_index]
    Xdata2 = xdata[mid_index:]
    xdata_fitted = np.mean([Xdata2[::-1], Xdata1], axis=0)
    xdata_fitted = pd.Series([*xdata_fitted, *xdata_fitted[::-1]])
    if x_unit:
        xdata_fitted.unit = x_unit

    #xdata uncertainty = half step size
    half_step_size = np.mean(np.abs(np.diff(xdata_fitted))) / 2
    #convert to pd.Series for consistency
    xdata_fitted_err = pd.Series([half_step_size] * len(xdata_fitted))
    if x_unit:
        xdata_fitted_err.unit = x_unit

    fitted_data = {
        'xdata': xdata_fitted, 
        'ydata': ydata_fitted,
        'xdata_err': xdata_fitted_err,
        'ydata_err': ydata_fitted_err,
    }

    # tan_hyseval_params orders the fitted values by 1. increasing HEB and 2. by decreasing HC
    # So for the decreasing branch, the field numbers correspond to the switching behavior from 
    # left to the right. Unless for the unlikely but possible event a very large coercive field 
    # strength with a more higher EB field for the second loop
    params = arctan_hyseval_params(result, xdata, ydata, sat_cond)

    return fitted_data, params, result

def arctan_hyseval_params(result, xdata, ydata, sat_cond=0.95):
    """
    Analytical calculation of important parameters of a hysteresis loop fitted with a tanh function
    (see tan_hyseval and tan_hys for details).
    
    Parameters
    ----------
    result : lmfit.model.ModelResult
        Result of the fit of the tanh function to the hysteresis loop.
    xdata : list or np.ndarray
        List of externally applied field strengths H.
    ydata : list or np.ndarray
        List of magnetization values M.
    sat_cond : float, optional
        Saturation condition for the tanh function. Default is 0.95.
    
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
    """
    
    # Check the length of the params dictionary. The following calculation should be performed n times for every 4*n + 1 parameters.
    if len(result.params) % 4 != 1:
        raise ValueError("The parameter dictionary does not contain the correct number of parameters.")
    
    nr_of_calculations = (len(result.params) - 1) // 4
    
    #xdata uncertainty = half step size
    half_step_size = np.mean(np.abs(np.diff(xdata))) / 2
    
    params = {}
    
    for i in range(1, nr_of_calculations + 1):
        # Assign the parameters a, b, c, d, e to the corresponding values of the current calculation
        a = result.params['a'].value # offset, equal for all calculations
        a_err = safe_stderr(result.params['a'].stderr)
        
        b = result.params[f'b_{i}'].value # amplitude
        b_err = safe_stderr(result.params[f'b_{i}'].stderr)
        
        c = result.params[f'c_{i}'].value # steepness
        c_err = safe_stderr(result.params[f'c_{i}'].stderr)
        
        d = result.params[f'd_{i}'].value # exchange bias field
        d_err = safe_stderr(result.params[f'd_{i}'].stderr)
        
        e = result.params[f'e_{i}'].value # coercive field
        e_err = safe_stderr(result.params[f'e_{i}'].stderr)
        
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

        # calculate saturation field strength (at 0.95*max(arctan(x))), ignore a as it is a global offset and does not influence the max of the spread
        # f(x) = a + b * 2/np.pi * np.arctan(c * (x - d +- e)); a = 0
        # arctan approaches maximum value of pi/2 so max of def arctan 
        # is a + b * 2/np.pi * np.pi/2 = a + b = b for a = 0
        # so 0.95 * +- b = b * 2/np.pi * arctan ( c * (x_sat - d +- e))
        # +- 0.95 * b = b * 2/np.pi * arctan ( c * (x_sat - d +- e))
        # +- 0.95 = 2/np.pi * arctan ( c * (x_sat - d +- e))
        # np.pi/2 * ( +- 0.95) = arctan ( c * (x_sat - d +- e))

        # tan(np.pi/2 * ( +- 0.95)) = c * (x_sat - d +- e)
        # tan(np.pi/2 * ( +- 0.95)) / c = x_sat - d +- e
        
        # tan(np.pi/2 * ( +- 0.95)) / c + d -+ e = x_sat

        # descending branch (neg. saturation field, positive HC shift): x_sat_desc = tan(np.pi/2 * ( - 0.95)) / c + d + e
        # ascending branch (pos. saturation field, negative HC shift): x_sat_asc = tan(np.pi/2 * ( + 0.95)) / c + d - e

        # x_sat error is equal for both branches
        
        # dx_sat_dc = np.abs( -tan(np.pi/2 * ( +- 0.95)) / c**2 * c_err )
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
        params['d' + key] = safe_stderr(result.params[key].stderr)
        
    if nr_of_calculations > 1:
        # Sort the params dictionary by 1. increasing HEB field and 2. decreasing coercive field strength
        HEBs = [params['HEB_' + str(i)] for i in range(1, nr_of_calculations + 1)]
        HCs = [params['HC_' + str(i)] for i in range(1, nr_of_calculations + 1)]
        sorted_indices = np.lexsort(([-hc for hc in HCs], HEBs)) # sort by HEB ascending and then by HC descending
        sorted_params = {}
        for key in params.keys():
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