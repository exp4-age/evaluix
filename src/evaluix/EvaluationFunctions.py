'''
This Python module is intended to provide all mathematical functions for Evaluix2
It is written blockwise for maintenance:
    1. Import all required external modules (numpy, pandas, scipy and lmfit)
    2. Basic functions
    3. Figure of Merit (FOM) functions
    4. Data manipulation functions
    5. Evaluation functions
        5.1 Hysteresis
        5.2 SRIM
        5.3 AFM
        5.4 Kerr?
    6. Specific functions
    
All functions are given with complete descriptions either visible in this 
document or function-specific callable via help(function)
'''
###############################################################################        
# 1. Import necessary modules
###############################################################################
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.integrate import quad
from typing import Union
from lmfit import Model, Parameters

#%%
###############################################################################        
# 2. Basic (Fit) Functions
###############################################################################
def linear(xdata: Union[float, int, list, pd.DataFrame, pd.Series, np.ndarray], a: float, b: float):
    """
    Calculate a linear function with a given slope `a` and offset `b`.

    This function is useful for fitting a linear function to data, such as removing linear slopes in the saturation regions of a hysteresis loop.

    Parameters
    ----------
    xdata : float, int, list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        Input value(s) (typically named x in functions).
    a : float
        Slope of the linear function.
    b : float
        Constant offset of the linear function.

    Returns
    -------
    ydata : numpy.ndarray
        Calculated value(s) of the linear function.

    Raises
    ------
    ValueError
        If `xdata` is not of a supported type.
    """
    # check xdata format
    if not isinstance(xdata, (int, float, list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'xdata must be a pandas dataframe, list, numpy array or int/float, not {type(xdata)}')

    xdata = np.asarray(xdata)
    return a * xdata + b

def polynomial(xdata: Union[float, int, list, pd.DataFrame, pd.Series, np.ndarray], *args: Union[float, list]):
    """
    Fit a polynomial to a dataset. The number of arguments determines the order of the polynomial. If orders should be skipped, use 0 as input.

    Parameters
    ----------
    xdata : float, int, list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        Input value(s) (typically named x in functions).
    *args : float or list
        Coefficients of the polynomial. The highest order of the fit is determined by the number of given arguments.
        For example, f(xdata) = args[0] * xdata ** (len(args) - 1) + args[1] * xdata ** (len(args) - 2) + ...

    Returns
    -------
    ydata : numpy.ndarray
        Calculated value(s) of the polynomial.

    Raises
    ------
    ValueError
        If `xdata` is not of a supported type.

    Examples
    --------
    >>> polynomial([1, 2, 3], 1, 0, -1)
    array([0., 1., 8.])
    """
    # check xdata format
    if not isinstance(xdata, (int, float, list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'xdata must be a pandas dataframe, list, numpy array or int/float, not {type(xdata)}')

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
    Compute the arctan function with given parameters.
    f(x) = a + b * arctan(c * (x - d + e))

    This function calculates the arctan function with the specified parameters, which can be used for branches of a hysteresis loop.

    Parameters
    ----------
    xdata : float, int, list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        Input value(s) (typically named x in functions).
    a : float
        Constant offset.
    b : float
        Coefficient/amplitude of arctan.
    c : float
        Coefficient of argument (xdata) inside arctan, mainly steepness.
    d : float
        Constant offset in xdata.
    e : float
        Branch dependent (in sign) offset in xdata.

    Returns
    -------
    ydata : numpy.ndarray
        Calculated value(s) of the arctan function.

    Examples
    --------
    >>> arctan([1, 2, 3], 1.0, 2.0, 3.0, 4.0, 5.0)
    array([1.0 + 2.0 * np.arctan(3.0 * (1 - 4 + 5)),
           1.0 + 2.0 * np.arctan(3.0 * (2 - 4 + 5)),
           1.0 + 2.0 * np.arctan(3.0 * (3 - 4 + 5))])
    """
    xdata = np.asarray(xdata)
    return a + b * np.arctan(c * (xdata - d + e))

def double_arctan(
    xdata: Union[float, int, list, pd.DataFrame, pd.Series, np.ndarray],
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    g: float,
    h: float,
    i: float,
):
    """
    Compute a double arctan function for one branch of a double hysteresis loop.

    This function takes a list or a single value of xdata and performs two arctan functions corresponding to one
    branch of a double hysteresis loop. The function is defined as:
    f(x) = 2*a + b * arctan(c * (x - d + e)) + f * arctan(g * (x - h + i))
    with d being a common offset (HEB) and e being a branch dependent offset (HC).
    The factor 2 in front of a is due to the fact that the function arctan is applied twice. It has no physical meaning,
    beside that the parameter a is now just half of the actual vertical offset.

    Parameters
    ----------
    xdata : float, int, list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        Input value(s) (typically named x in functions).
    a : float
        Constant offset. As the function is applied twice, the offset is doubled.
    b : float
        Coefficient/amplitude of the first arctan (Hys1).
    c : float
        Coefficient of the argument (xdata) inside the first arctan, mainly steepness (Hys1).
    d : float
        Constant offset in xdata for the first arctan (Hys1).
    e : float
        Branch dependent (in sign) offset in xdata for the first arctan (Hys1).
    f : float
        Coefficient/amplitude of the second arctan (Hys2).
    g : float
        Coefficient of the argument (xdata) inside the second arctan, mainly steepness (Hys2).
    h : float
        Constant offset in xdata for the second arctan (Hys2).
    i : float
        Branch dependent (in sign) offset in xdata for the second arctan (Hys2).

    Returns
    -------
    ydata : numpy.ndarray
        Calculated value(s) of the double arctan function.

    Examples
    --------
    >>> double_arctan([1, 2, 3], 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    array([calculated_value1, calculated_value2, calculated_value3])
    """
    xdata = np.asarray(xdata)
    return arctan(xdata, a, b, c, d, e) + arctan(xdata, a, f, g, h, i)

def tan_hys(
    xdata: Union[float, int, list, pd.DataFrame, pd.Series, np.ndarray],
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
):
    """
    Compute the arctan function for the two branches of a single hysteresis loop.

    This function takes a list or a single value of xdata and performs two arctan functions corresponding to the 
    two branches of a hysteresis loop. The function is defined as:
    y = a + b * arctan(c * (x - d + e)); for increasing x and
      = a + b * arctan(c * (x - d - e)); for decreasing x
    with d being a common horizontal offset (HEB) and e being a branch dependent offset (HC).

    Parameters
    ----------
    xdata : float, int, list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        Input value(s) (typically named x in functions).
    a : float
        Constant offset.
    b : float
        Coefficient/amplitude of arctan.
    c : float
        Coefficient of argument (xdata) inside arctan, mainly steepness.
    d : float
        Constant offset in xdata (e.g., HEB).
    e : float
        Branch dependent (in sign) offset in xdata (e.g., HC).

    Returns
    -------
    tuple
        If xdata is a single value, returns a tuple of (mean, ydata1, ydata2).
        If xdata is a list or array, returns the concatenated ydata for both branches.

    Examples
    --------
    >>> tan_hys(1.0, 1.0, 2.0, 3.0, 4.0, 5.0) # for single values
    (mean_value, ydata1, ydata2)
    >>> tan_hys([1, 2, 3, 4, 5, 6], 1.0, 2.0, 3.0, 4.0, 5.0) # for lists
    array([ydata1_values, ydata2_values])
    """
    # if arctan of a single value is wanted. Return the mean of both branches
    # as well as the individual branches
    if isinstance(xdata, (int, float)): # for calculating single values
        ydata1 = arctan(xdata, a, b, c, d, e)
        ydata2 = arctan(xdata, a, b, c, d, -e)
        return np.mean([np.abs(ydata1), np.abs(ydata2)]), ydata1, ydata2
    
    elif isinstance(xdata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        #if xdata is given as a list (hysteresis), split it correspondingly into two branches
        xdata = np.asarray(xdata)

        # Check if the length of xdata is odd, i.e.the center point contributes to both branches.
        # Duplicate the center point in this case so that both branches are equally long.
        if len(xdata) % 2 != 0:
            center_index = len(xdata) // 2
            xdata = np.insert(xdata, center_index, xdata[center_index])
            ydata = np.insert(ydata, center_index, ydata[center_index])

        Xdata = np.split(xdata, 2)
        ydata1 = arctan(Xdata[0], a, b, c, d, e)
        ydata2 = arctan(Xdata[1], a, b, c, d, -e)
        return np.append(ydata1, ydata2)

def double_tan_hys(
    xdata: Union[float, int, list, pd.DataFrame, pd.Series, np.ndarray],
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    g: float,
    h: float,
    i: float,
):
    """
    Compute the double arctan function for the two branches of a double hysteresis loop.

    This function takes a list or a single value of xdata and performs four arctan functions corresponding 
    to two branches of a double hysteresis loop. For each appearing hysteresis, the two branch parts are 
    connected by the tan_hys() function. Therefore, this function is basically applied twice.

    A small if statement is used to make sure that the second hysteresis starts after (higher fields) the first one.
    Thus f-i belong to the hysteresis loop at a larger Exchange bias field strength.

    Parameters
    ----------
    xdata : float, int, list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        Input value(s) (typically named x in functions).
    a : float
        Constant offset. As the function is applied twice, the offset is doubled.
    b : float
        Coefficient/amplitude of the first arctan (Hys1).
    c : float
        Coefficient of the argument (xdata) inside the first arctan, mainly steepness (Hys1).
    d : float
        Constant offset in xdata for the first arctan (Hys1).
    e : float
        Branch dependent (in sign) offset in xdata for the first arctan (Hys1).
    f : float
        Coefficient/amplitude of the second arctan (Hys2).
    g : float
        Coefficient of the argument (xdata) inside the second arctan, mainly steepness (Hys2).
    h : float
        Constant offset in xdata for the second arctan (Hys2).
    i : float
        Branch dependent (in sign) offset in xdata for the second arctan (Hys2).

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
    if h < d: # Make sure that the second hysteresis starts after (higher fields) the first one
        h = d + 1e-4  # Adjust h to be slightly greater than d

    # if double arctan of a single value is wanted. Return the mean of both branches
    # as well as the individual branches
    if isinstance(xdata, (int, float)): # for calculating single values
        ydata1 =  arctan(xdata, a, b, c, d, e) + arctan(xdata, a, f, g, h, i)
        ydata2 =  arctan(xdata, a, b, c, d, -e) + arctan(xdata, a, f, g, h, -i)
        return np.mean([np.abs(ydata1), np.abs(ydata2)]), ydata1, ydata2
    
    elif isinstance(xdata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        # forward to tan_hys() function and its logic
        return tan_hys(xdata, a, b, c, d, e) + tan_hys(xdata, a, f, g, h, i)

#%%
###############################################################################        
# 3. FOM (Figure of Merit) functions
###############################################################################

def chisquared_lmfit(
    ydata: Union[list, pd.DataFrame, pd.Series, np.ndarray], 
    model: Union[list, pd.DataFrame, pd.Series, np.ndarray],
    weights: Union[list, pd.DataFrame, pd.Series, np.ndarray] = None):
    """
    Calculate the chi-squared value of a fit function to a dataset.

    The chi-squared value is a measure of how well the observed outcomes are 
    replicated by the model. It is the sum of the squared differences between 
    the observed and predicted values, divided by the predicted values or weights.

    Parameters
    ----------
    ydata : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        Observed output value(s) of the dataset (typically named y in functions).
    model : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        Model output.
    weights : list, numpy.ndarray, pandas.DataFrame, or pandas.Series, optional
        Weights for each data point. If not provided, the model values are used.

    Returns
    -------
    chi_squared : float
        Chi-squared value.

    Examples
    --------
    >>> chisquared_lmfit([1, 2, 3], [1.1, 1.9, 3.2])
    0.02
    >>> chisquared_lmfit([1, 2, 3], [1.1, 1.9, 3.2], [1, 1, 1])
    0.02
    """
    if weights is not None:
        # calculate the chi-squared value with weights
        chi_squared = np.sum((ydata - model)**2 / weights)
    else:
        # calculate the chi-squared value
        chi_squared = np.sum((ydata - model)**2 / model)
    
    return chi_squared

#%%
###############################################################################        
# 4. Data Manipulation
###############################################################################

def smoothing1d(ydata: pd.DataFrame, smoothing_fct: str = 'savgol', window_length: int = 5, sigma_or_polyorder: int = 2):
    """
    Category: Data Manipulation

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
        return smoothed.fillna(method='nearest')
    elif str(smoothing_fct) == 'median':
        smoothed = ydata.rolling(window_length, center=True).median()
        return smoothed.fillna(method='nearest')
    elif str(smoothing_fct) == 'None' or smoothing_fct is None:
        smoothed = ydata.rolling(window_length, center=True).mean()
        return smoothed.fillna(method='nearest')
    else:
        raise ValueError(f'Smoothing function not recognized. Please choose between savgol, gaussian, median, or None, not {smoothing_fct}')
    
def del_outliers(
    ydata: Union[list, pd.DataFrame, pd.Series, np.ndarray], 
    threshold: float = 2, 
    neighbours: int = 5):
    """
    Category: Data Manipulation

    Remove outliers from the dataset based on a specified threshold and number of neighbours.

    An outlier is defined as a point that differs from the mean of all data points by more than 
    `threshold` times the mean absolute difference (MAD). Additionally, it must differ from the 
    mean of the closest `neighbours` by more than 0.5 * `threshold` * MAD or 
    `threshold` * MAD(neighbours).

    The point is then replaced by the linear interpolation of its neighbours.

    Parameters
    ----------
    ydata : list, numpy.ndarray, pandas.DataFrame, or pandas.Series
        Output value(s) of the dataset (typically named y in functions).
    threshold : float, optional
        Threshold for the difference of a point to the mean of all points. The default is 2.
    neighbours : int, optional
        Number of neighbours to be taken into account for the second mean calculation as well as 
        for the linear interpolation. The default is 5.

    Returns
    -------
    np.ndarray
        Array with removed outliers if there were any. Does not change the data if no outliers were present.

    Raises
    ------
    ValueError
        If `ydata` is not of a supported type.

    Examples
    --------
    >>> del_outliers([1, 2, 3, 100, 5, 6, 7])
    array([1, 2, 3, 4, 5, 6, 7])
    """
    if not isinstance(ydata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'ydata must be a pandas dataframe, list or numpy array, not {type(ydata)}')
    
    ydata = np.asarray(ydata).copy()
    
    # calculate mean and mean absolute difference (mad) of all points
    mean = np.mean(ydata)
    mad = np.mean(np.abs(ydata - mean))
    
    # calculate mean and standard deviation of the closest neighbours
    mean_neighbours = np.zeros_like(ydata)
    mad_neighbours = np.zeros_like(ydata)

    for i in range(len(ydata)):
        start = max(0, i - neighbours) #left neighbours, make sure to not go below 0
        end = min(i + neighbours, len(ydata)) #right neighbours, make sure to not go above the last index
        neighbours_slice = ydata[start:end] #slice of neighbours
        mean_neighbours[i] = np.mean(neighbours_slice) #mean of neighbours
        mad_neighbours[i] = np.mean(np.abs(neighbours_slice - mean_neighbours[i])) #mad of neighbours

    # Identify outliers
    outliers =  (np.abs(ydata - mean) > threshold * mad) & \
                (np.abs(ydata - mean_neighbours) > 0.5 * threshold * mad_neighbours) | \
                (np.abs(ydata - mean_neighbours) > threshold * mad_neighbours)
    outlier_indices = np.where(outliers)[0]
    # Outliers are points which differ from the mean of all points by more than threshold * mad
    # and from the mean of the closest neighbours by more than 0.5 * threshold * mad_neighbours
    # or from the mean of the closest neighbours by more than threshold * mad_neighbours
    
    # Replace outliers with the interpolated mean value of their neighbours
    for outlier in outlier_indices:
        if outlier:
            start = max(0, outlier - neighbours) #left neighbours, make sure to not go below 0
            end = min(outlier + neighbours + 1, len(ydata)) #right neighbours, make sure to not go above the last index
            neighbours_slice = np.concatenate([ydata[start:outlier], ydata[outlier+1:end]]) #exclude outlier
            ydata[outlier] = np.mean(neighbours_slice) #replace outlier with mean of neighbours
    
    return ydata

def rmv_opening(ydata: Union[list, pd.DataFrame, pd.Series, np.ndarray], sat_region: float = 0.05):
    """
    Category: Data Manipulation

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
    
    # check if difference is below the sum of both standard deviations (noise level) and therefore no
    # significant opening is present
    if np.abs(mean_diff) < np.std(ydata[:sat_points]) + np.std(ydata[-sat_points:]):
        return ydata

    # calculate slope of opening with respect to the length of the hysteresis
    # ignore the outermost 2*0.5*sat_region of points, since the diff represents the mean at the outermost 0.5*sat_region positions
    opening_slope = - mean_diff / (len(ydata) - sat_points)
    
    # subtract slope from ydata. Reminder slope is in time/point number and not in xdata/field strength
    ydata -= opening_slope * np.arange(len(ydata))
    
    return ydata

def slope_correction(
    xdata: Union[list, pd.DataFrame, pd.Series, np.ndarray],
    ydata: Union[list, pd.DataFrame, pd.Series, np.ndarray],
    sat_region: float = 0.1,
    noise_threshold: float = 3,
    branch_difference: float = 0.3):
    """
    Category: Data Manipulation

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
    noise_level = np.mean([np.std(residuals_upper), np.std(residuals_lower)])

    # Check if the slope (effect over the field range) is insignificant (below the noise level) and do nothing if it is
    if np.abs(slope) * (np.max(xdata) - np.min(xdata)) < noise_threshold * noise_level or np.abs(1 - popt1[0]/popt2[0]) > branch_difference:
        return ydata

    # otherwise return subtracted/corrected magnetization
    return ydata - slope * xdata

def hys_norm(
    xdata: Union[list, pd.DataFrame, pd.Series, np.ndarray],
    ydata: Union[list, pd.DataFrame, pd.Series, np.ndarray],
    sat_region: float = 0.1):
    """
    Category: Data Manipulation

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
        return (ydata - y_bias) / norm

def hys_center(
    xdata: Union[list, pd.DataFrame, pd.Series, np.ndarray],
    ydata: Union[list, pd.DataFrame, pd.Series, np.ndarray],
    sat_region: float = 0.1,
    normalize: bool = False):
    """
    Category: Data Manipulation

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

    return xdata, ydata

#%%
###############################################################################        
# 5. Data Evaluation
###############################################################################

def x_sect(xdata: pd.Series, ydata: pd.Series, steepness_for_fit: bool = False):
    """
    Category: Data Evaluation

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
    >>> x_sect(pd.Series([1, 2, 3]), pd.Series([-1, 0, 1]), steepness_for_fit=True)
    (2.0, 0.0, 1.0)
    """
    
    if len(xdata) != len(ydata):
        raise ValueError('xdata and ydata must have the same length')
    
    if not isinstance(xdata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'xdata must be a pandas dataframe, list or numpy array, not {type(xdata)}')
    
    if not isinstance(ydata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'ydata must be a pandas dataframe, list or numpy array, not {type(ydata)}')
    
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    
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
                if a != 0:
                    b = ydata[i-1] - a * xdata[i-1]
                    intersect = -b / a
                    intersect_err = max(np.abs(intersect - xdata[i]), np.abs(intersect - xdata[i-1]))

    if steepness_for_fit:
        return intersect, intersect_err, a
    else:
        return intersect, intersect_err
    
def y_sect(xdata: pd.Series, ydata: pd.Series, HEB: float = 0):
    """
    Category: Data Evaluation

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
    HEB : float, optional
        Exchange bias field strength. The default is 0.

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
    >>> y_sect(pd.Series([1, 2, 3]), pd.Series([-1, 0, 1]), HEB=1)
    (0.0, 0.0)
    """
    
    if len(xdata) != len(ydata):
        raise ValueError('xdata and ydata must have the same length')
    
    if not isinstance(xdata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'xdata must be a pandas dataframe, list or numpy array, not {type(xdata)}')
    
    if not isinstance(ydata, (list, pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(f'ydata must be a pandas dataframe, list or numpy array, not {type(ydata)}')
    
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    
    # Ensure that the xdata list starts with a negative value (from left to right)
    # The sign change of HEB ensures the correct sign for each branch
    if list(xdata)[0] > 0.0:
        HEB = -HEB
        ydata = np.flipud(ydata)
        xdata = np.flipud(xdata)

    # Initialize variables
    intersect = 0.0
    intersect_err = 0.0

    # Check number of points with > HEB
    if len(np.where(xdata >= HEB)[0]) > 0:
        # Take the first point (i) with > HEB and linearly interpolate slope a 
        # at intersection with one point before (i-1)
        i = np.where(xdata >= HEB)[0][0]
        a = (list(ydata)[i] - list(ydata)[i-1]) / (list(xdata)[i] - list(xdata)[i-1])
        if not (a == 0):
            # Assume linearly interpolated intersection, calculate offset b and
            # transform it to intersect param (shift towards HEB value)
            b = list(ydata)[i-1] - a * list(xdata)[i-1]
            intersect = b + a * HEB
            # Calculate maximum deviation of utilized points to the 
            # intersection as uncertainty
            intersect_err = max(
                [np.abs(intersect - list(ydata)[i]), 
                 np.abs(intersect - list(ydata)[i-1])]
                )
            
    # Intersection and intersection uncertainty
    return intersect, intersect_err
    
def num_derivative(xdata: pd.Series, ydata: pd.Series):
    """
    Category: Data Manipulation

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
    Category: Data Manipulation

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

def lin_hyseval(xdata, ydata, offset: float = 0.0, steepness_for_fit: bool = False):
    """
    Category: Data Evaluation

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
    ydata = np.asarray(ydata) - offset
    
    # Check if the length of xdata is odd, i.e., the center point contributes to both branches.
    # Duplicate the center point in this case so that both branches are equally long.
    if len(xdata) % 2 != 0:
        center_index = len(xdata) // 2
        xdata = np.insert(xdata, center_index, xdata[center_index])
        ydata = np.insert(ydata, center_index, ydata[center_index])
    
    # Obtain intersections as coercive fields with the x_sect function
    # branch-dependently
    HC1, dHC1, a1 = x_sect(np.split(xdata, 2)[0], np.split(ydata, 2)[0], steepness_for_fit=True) # first branch
    HC2, dHC2, a2 = x_sect(np.split(xdata, 2)[1], np.split(ydata, 2)[1], steepness_for_fit=True) # second branch
    
    half_step_size = np.mean(np.abs(np.diff(xdata))) / 2
    
    # EB field as average of coercive fields/intersects
    HEB = (HC1 + HC2) / 2
    dHEB = (dHC1 + dHC2) / 2 + half_step_size # uncertainty via propagation of uncertainty
    # Coercive field as half of the distance between the two intersections
    HC = np.abs((HC1 - HC2) / 2)
    dHC = (dHC1 + dHC2) / 2 + half_step_size # uncertainty via propagation of uncertainty
    
    # Remanence at zero field strength
    MR1, dMR1 = y_sect(np.split(xdata, 2)[0], np.split(ydata, 2)[0], 0)
    MR2, dMR2 = y_sect(np.split(xdata, 2)[1], np.split(ydata, 2)[1], 0)
    # Average of both branches
    MR = ((np.abs(MR1) + np.abs(MR2)) / 2, MR1, MR2)
    dMR = (dMR1 + dMR2) / 2 # uncertainty via propagation of uncertainty

    # Magnetization at the exchange bias field
    MHEB1, dMHEB1 = y_sect(np.split(xdata, 2)[0], np.split(ydata, 2)[0], HEB)
    MHEB2, dMHEB2 = y_sect(np.split(xdata, 2)[1], np.split(ydata, 2)[1], HEB)
    # Average of both branches
    MHEB = (np.abs(MHEB1) + np.abs(MHEB2)) / 2
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

def tan_hyseval(xdata, ydata, sat_cond: float = 0.95):
    """
    Category: Data Evaluation

    Fits a hysteresis loop with an arctan function to extract several parameters
    including the exchange bias field, coercive field strength, remanence, saturation
    magnetization, saturation field strength, the slopes at the intersection with the
    x-axis (HC field) and the EB field, a possible offset, the enclosed area and 
    enclosed angle of the hysteresis loop, and the rectangularity of the hysteresis.
    For all parameters, the uncertainty is calculated via propagation of uncertainty.

    It also provides the fitted ydata and its uncertainty as well as the lmfit result.

    Note: For some uncertainties, the error propagation is not fully correct and rather
    estimated. The function is not suitable for double hysteresis loops.
    
    Parameters
    ----------
    xdata : list or np.ndarray
        List of externally applied field strengths H.
    ydata : list or np.ndarray
        List of magnetization values M.
    sat_cond : float, optional
        Saturation condition for the tanh function. Default is 0.95.

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
    
    if hasattr(xdata, 'unit'):
        x_unit = xdata.unit
    else:
        x_unit = None
    if hasattr(ydata, 'unit'):
        y_unit = ydata.unit
    else:
        y_unit = None
    
    # quick linear calculation to determine initial guesses for the exchange bias and the coercive field strength
    LIN = lin_hyseval(xdata, ydata, steepness_for_fit=True)
    HEB_tmp, HC_tmp = LIN['HEB'], LIN['HC']
    slope = (LIN['a1'] + LIN['a2']) # No average because the fit works better with a steeper initial slope
    
    # Create a model from the function
    model = Model(tan_hys)

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
    params.add('b', value=(np.max(ydata) - np.min(ydata))/np.pi) # amplitude
    params.add('c', value=slope / params['b'].value, min=0) # steepness
    params.add('d', value=HEB_tmp, min=np.min(xdata), max=np.max(xdata)) # exchange bias field
    params.add('e', value=HC_tmp, min=0, max=(np.max(xdata) - np.min(xdata)) / 2) # coercive field

    # Fit the model to the data
    result = model.fit(ydata, params, calc_covar=True, method='leastsq', fit_kws={'reduce_fcn': chisquared_lmfit}, xdata=xdata,)

    # Calculate fitted magnetization values
    ydata = result.best_fit
    #convert to pd.Series for consistency
    ydata = pd.Series(ydata)
    if y_unit:
        ydata.unit = y_unit

    # calculate the 3 sigma uncertainty of the fit
    ydata_err = result.eval_uncertainty(sigma=3)
    # convert to pd.Series for consistency
    ydata_err = pd.Series(ydata_err)
    if y_unit:
        ydata_err.unit = y_unit

    # create a copy of the averaged xdata (branch 1 and 2) with which the results are
    # displayed. Only important for plotting the area of the hysteresis loop. As long as
    # the xdata is symmetric, the average is symmetric as well.
    xdata = np.mean([np.split(xdata, 2)[1][::-1], np.split(xdata, 2)[0]], axis=0)
    xdata = pd.Series([*xdata, *xdata[::-1]])
    if x_unit:
        xdata.unit = x_unit

    #xdata uncertainty = half of average step size
    #TODO: For Kerr and VSM the step size may change, so this should be calculated from the data
    #TODO: np.diff without mean? In that case: HC/HEB error should be calculated at their respective positions
    #TODO: And I have to make sure that np.diff has the same length as xdata
    half_step_size = np.mean(np.abs(np.diff(xdata))) / 2
    #convert to pd.Series for consistency
    xdata_err = pd.Series([half_step_size] * len(xdata))
    if x_unit:
        xdata_err.unit = x_unit

    # Extract wanted values from the fits optimized params
    HC, dHC = result.params['e'].value, safe_stderr(result.params['e'].stderr) + half_step_size
    HEB, dHEB = result.params['d'].value, safe_stderr(result.params['d'].stderr) + half_step_size
    MR = tan_hys(float(0), *result.params.valuesdict().values())
    dMR = safe_stderr(result.params['a'].stderr) + safe_stderr(result.params['b'].stderr)
    MHEB = tan_hys(float(HEB), *result.params.valuesdict().values())
    dMHEB = dMR

    slope_atHC = result.params['b'].value * result.params['c'].value
    dslope_atHC = safe_stderr(result.params['b'].stderr) * result.params['c'].value + safe_stderr(result.params['c'].stderr) * result.params['b'].value
    slope_atHEB = (result.params['b'].value * result.params['c'].value) / (1 + (result.params['c'].value * result.params['e'].value)**2)
    dslope_atHEB = dslope_atHC * 0.5 #TODO: This is not true, but I don't want to invest time in this right now

    # Calculate the rectangularity of the hysteresis loop by interpolating a parallelogram with the slopes at HC and HEB. The closer these angles are to 90°, the more rectangular the hysteresis loop is.
    # Normalize the slope to make it independent on the scaling for MS and Hext: divide 2 * HC by the amplitude of the hysteresis loop
    slope_normalization = 2 * HC / result.params['b'].value

    # angles of the slopes at HC and HEB
    angle_atHC = np.arctan(slope_atHC * slope_normalization)
    dangle_atHC = np.arctan(dslope_atHC * slope_normalization) #TODO: Confirm if this is correct
    angle_atHEB = np.arctan(slope_atHEB * slope_normalization)
    dangle_atHEB = np.arctan(dslope_atHEB * slope_normalization)
    
    # angles at the intersection points. Alpha1 is the angle enclosed between the slopes at HC and HEB, i.e. the angle at the lower left corner of the parallelogram. Alpha2 is the angle at the upper right corner but their sum is always 180°, so it does not need to be calculated.
    alpha = angle_atHC - angle_atHEB
    dalpha = np.sqrt(dangle_atHC**2 + dangle_atHEB**2)
    rectangularity = np.sin(alpha) # from 0 to 1, 1 is a perfect rectangle at alpha1 = 90°
    drectangularity = np.abs(np.cos(alpha)) * dalpha
    # rectangularity2 = 1 - (np.abs(alpha1 - np.pi/2) + np.abs(alpha1 + np.pi/2)) / np.pi # less sensitive therefore not used

    # calculate saturation field strength (at 0.95*max(arctan(x)))
    # f(x) = a  + b * np.arctan(c * (x - d +- e))
    # arctan approaches maximum value of pi/2 so max of def arctan 
    # is a + b * pi/2 
    # so 0.95 * (a + b * pi/2) = a + b * arctan ( c * (x_sat - d +- e))
    # 0.95 * pi/2 - 0.05 * a/b = arctan (c * (x_sat - d +- e))
    # tan(0.95 * pi/2  - 0.05 * a/b) = c * (x_sat - d +- e)
    # x_sat = tan(0.95 * pi/2 - 0.05 * a/b) / c + d -+ e
    # x_sat_err = 1 / (c * cos(0.99 * pi/2)^2) * c_err + d_err + e_err ?

    saturation_field1 = np.tan(sat_cond * np.pi / 2 - (1-sat_cond) * result.params['a'].value / result.params['b'].value) / result.params['c'].value + HEB - HC
    saturation_field2 = saturation_field1 + 2 * HC # add 2 * HC to get the saturation field in the other direction
    saturation_field3 = np.tan(-sat_cond * np.pi / 2 - (1-sat_cond) * result.params['a'].value / result.params['b'].value) / result.params['c'].value + HEB - HC
    saturation_field4 = saturation_field3 + 2 * HC # add 2 * HC to get the saturation field in the other direction
    saturation_fields = (np.min([saturation_field1, saturation_field2, saturation_field3, saturation_field4]), 
                         np.max([saturation_field1, saturation_field2, saturation_field3, saturation_field4]))
    dsaturation_fields = (half_step_size, half_step_size) #TODO: This is only partly true, but I don't want to invest time in this right now

    integral_args = result.params.valuesdict()
    # change param a and e
    integral_args['a'] -= np.min(ydata)
    # Integral of the area in the loop = left branch integral - right branch integral
    integral_leftbranch, integral_leftbranch_err = quad(arctan, np.min(xdata), np.max(xdata), args=tuple(integral_args.values()))
    integral_args['e'] *= -1
    integral_rightbranch, integral_rightbranch_err = quad(arctan, np.min(xdata), np.max(xdata), args=tuple(integral_args.values()))
    # Area of the hysteresis loop
    integral = integral_leftbranch - integral_rightbranch
    integral_err = np.sqrt(integral_leftbranch_err**2 + integral_rightbranch_err**2)

    fitted_data = {
        'xdata': xdata, 
        'ydata': ydata,
        'xdata_err': xdata_err,
        'ydata_err': ydata_err,
    }
    params = {
        'r_squared': result.rsquared,
        'HEB': HEB, 
        'dHEB': dHEB,
        'HC': HC, 
        'dHC': dHC,
        'MR': MR,
        'dMR': dMR,
        'MHEB': MHEB,
        'dMHEB': dMHEB,
        'integral': integral,
        'dintegral': integral_err,
        'saturation_fields': saturation_fields,
        'dsaturation_fields': dsaturation_fields,

        'slope_atHC': slope_atHC,
        'dslope_atHC': dslope_atHC,
        'slope_atHEB': slope_atHEB,
        'dslope_atHEB': dslope_atHEB,
        'alpha': alpha,
        'dalpha': dalpha,
        'rectangularity': rectangularity,
        'drectangularity': drectangularity,

        'x_unit': x_unit,
        'y_unit': y_unit,
        }
    
    return fitted_data, params, result

def double_tan_hyseval(xdata, ydata, sat_cond: float = 0.95):
    """
    Category: Data Evaluation

    Fits a hysteresis loop with a double tanh function to extract several parameters
    including the exchange bias field, coercive field strength, remanence, saturation
    magnetization, saturation field strength, the slopes at the intersection with the
    x-axis and the EB field, the steepness of the tanh function, a possible offset, the
    enclosed area and angle of the hysteresis loop, and the rectangularity of the hysteresis.

    It also provides the fitted ydata and its uncertainty as well as the lmfit result.

    Note: For some uncertainties, the error propagation is not fully correct and rather
    estimated. The function is suitable for double hysteresis loops but not for single
    hysteretic loops due to overfitting.

    Parameters
    ----------
    xdata : list or np.ndarray
        List of externally applied field strengths H.
    ydata : list or np.ndarray
        List of magnetization values M.
    sat_cond : float, optional
        Saturation condition for the tanh function. Default is 0.95.

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
            - ratio : dict
                Dictionary containing:
                - HEB1/HEB2 : float
                    Ratio of exchange bias fields.
                - HC1/HC2 : float
                    Ratio of coercive fields.
                - A1/A2 : float
                    Ratio of amplitudes.
                - area1/area2 : float
                    Ratio of areas.
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
    LIN = lin_hyseval(xdata, ydata)
    HEB_tmp, HC_tmp = LIN['HEB'], LIN['HC']
    
    # Create a model from the function
    model = Model(double_tan_hys)

    # Define the parameters
    params = Parameters()
    params.add('a', value=0.0) # offset, 0 for normalized hysteresis with pos/neg Sat.
    params.add('b', value=np.max(ydata)/np.pi) # amplitude, half of the maximum of the hysteresis
    params.add('c', value=5.0) # steepness
    params.add('d', value=HEB_tmp - 0.2 * np.abs(np.min(xdata)), min=np.min(xdata), max=np.max(xdata)) # lower exchange bias field
    params.add('e', value=HC_tmp, min=0, max=np.max(xdata) - np.min(xdata)) # coercive field
    params.add('f', value=np.max(ydata)/np.pi) # amplitude, half of the maximum of the hysteresis
    params.add('g', value=5.0) # steepness
    # h has to be larger or equal to d
    params.add('h', value=HEB_tmp + 0.2 * np.abs(np.max(xdata)), min=np.min(xdata), max=np.max(xdata)) # upper exchange bias field
    params.add('i', value=HC_tmp, min=0, max=np.max(xdata) - np.min(xdata)) # coercive field

    # Fit the model to the data
    result = model.fit(ydata, params, calc_covar=True, method='leastsq', fit_kws={'reduce_fcn': chisquared_lmfit}, xdata=xdata,)

    # Calculate fitted magnetization values
    ydata = result.best_fit
    #convert to pd.Series for consistency
    ydata = pd.Series(ydata)
    if y_unit:
        ydata.unit = y_unit

    # calculate the 3 sigma uncertainty of the fit
    ydata_err = result.eval_uncertainty(sigma=3)
    # convert to pd.Series for consistency
    ydata_err = pd.Series(ydata_err)
    if y_unit:
        ydata_err.unit = y_unit

    # create a copy of the averaged xdata (branch 1 and 2) with which the results are
    # displayed
    xdata = np.mean([np.split(xdata, 2)[1][::-1], np.split(xdata, 2)[0]], axis=0)
    xdata = pd.Series([*xdata, *xdata[::-1]])
    if x_unit:
        xdata.unit = x_unit

    #xdata uncertainty = half step size
    half_step_size = np.mean(np.abs(np.diff(xdata))) / 2
    #convert to pd.Series for consistency
    xdata_err = pd.Series([half_step_size] * len(xdata))
    if x_unit:
        xdata_err.unit = x_unit

    # Extract wanted values from the fits optimized params
    HC1, dHC1 = result.params['e'].value, safe_stderr(result.params['e'].stderr) + half_step_size
    HEB1, dHEB1 = result.params['d'].value, safe_stderr(result.params['d'].stderr) + half_step_size
    HC2, dHC2 = result.params['i'].value, safe_stderr(result.params['i'].stderr) + half_step_size
    HEB2, dHEB2 = result.params['h'].value, safe_stderr(result.params['h'].stderr) + half_step_size

    MR = double_tan_hys(float(0), *result.params.valuesdict().values())
    dMR = safe_stderr(result.params['a'].stderr) + safe_stderr(result.params['b'].stderr) + safe_stderr(result.params['f'].stderr)
    MHEB1 = double_tan_hys(float(HEB1), *result.params.valuesdict().values())
    dMHEB1 = dMR
    MHEB2 = double_tan_hys(float(HEB2), *result.params.valuesdict().values())
    dMHEB2 = dMR

    slope_atHC1 = result.params['b'].value * result.params['c'].value
    dslope_atHC1 = safe_stderr(result.params['b'].stderr) * result.params['c'].value + safe_stderr(result.params['c'].stderr) * result.params['b'].value
    slope_atHEB1 = (result.params['b'].value * result.params['c'].value) / (1 + (result.params['c'].value * result.params['e'].value)**2)
    dslope_atHEB1 = dslope_atHC1 * 0.5 #TODO: This is not true, but I don't want to invest time in this right now
    slope_atHC2 = result.params['f'].value * result.params['g'].value
    dslope_atHC2 = safe_stderr(result.params['f'].stderr) * result.params['g'].value + safe_stderr(result.params['g'].stderr) * result.params['f'].value
    slope_atHEB2 = (result.params['f'].value * result.params['g'].value) / (1 + (result.params['g'].value * result.params['i'].value)**2)
    dslope_atHEB2 = dslope_atHC2 * 0.5 #TODO: This is not true, but I don't want to invest time in this right now

    # Calculate the rectangularity of the hysteresis loop by interpolating a parallelogram with the slopes at HC and HEB. The closer these angles are to 90°, the more rectangular the hysteresis loop is.
    # Normalize the slope to make it independent on the scaling for MS and Hext: divide 2 * HC by the amplitude of the hysteresis loop
    slope_normalization1 = 2 * HC1 / result.params['b'].value
    slope_normalization2 = 2 * HC2 / result.params['f'].value

    # angles of the slopes at HC and HEB
    angle_atHC1 = np.arctan(slope_atHC1 * slope_normalization1)
    dangle_atHC1 = np.arctan(dslope_atHC1 * slope_normalization1) #TODO: Confirm if this is correct
    angle_atHEB1 = np.arctan(slope_atHEB1 * slope_normalization1)
    dangle_atHEB1 = np.arctan(dslope_atHEB1 * slope_normalization1) #TODO: Confirm if this is correct
    angle_atHC2 = np.arctan(slope_atHC2 * slope_normalization2)
    dangle_atHC2 = np.arctan(dslope_atHC2 * slope_normalization2) #TODO: Confirm if this is correct
    angle_atHEB2 = np.arctan(slope_atHEB2 * slope_normalization2)
    dangle_atHEB2 = np.arctan(dslope_atHEB2 * slope_normalization2) #TODO: Confirm if this is correct
    
    # angles at the intersection points. Alpha1 is the angle enclosed between the slopes at HC and HEB, i.e. the angle at the lower left corner of the parallelogram. Alpha2 is the angle at the upper right corner but their sum is always 180°, so it does not need to be calculated.
    alpha1 = angle_atHC1 - angle_atHEB1
    dalpha1 = np.sqrt(dangle_atHC1**2 + dangle_atHEB1**2)
    rectangularity1 = np.sin(alpha1) # from 0 to 1, 1 is a perfect rectangle at alpha1 = 90°
    drectangularity1 = np.abs(np.cos(alpha1)) * dalpha1
    alpha2 = angle_atHC2 - angle_atHEB2
    dalpha2 = np.sqrt(dangle_atHC2**2 + dangle_atHEB2**2)
    rectangularity2 = np.sin(alpha2) # from 0 to 1, 1 is a perfect rectangle at alpha1 = 90°
    drectangularity2 = np.abs(np.cos(alpha2)) * dalpha2

    # calculate saturation field strength (at sat_cond*max(arctan(x))), sat_cond = 0.95
    # see tan_hyseval for calculation
    saturation_field1_1 = np.tan(sat_cond * np.pi / 2) / result.params['c'].value + HEB1 - HC1
    saturation_field1_2 = np.tan(sat_cond * np.pi / 2) / result.params['c'].value + HEB1 + HC1
    saturation_field1_3 = np.tan(-sat_cond * np.pi / 2) / result.params['c'].value + HEB1 - HC1
    saturation_field1_4 = np.tan(-sat_cond * np.pi / 2) / result.params['c'].value + HEB1 + HC1
    saturation_fields1 = (np.min([saturation_field1_1, saturation_field1_2, saturation_field1_3, saturation_field1_4]), 
                          np.max([saturation_field1_1, saturation_field1_2, saturation_field1_3, saturation_field1_4]))
    dsaturation_fields1 = (half_step_size, half_step_size) #TODO: This is only partly true, but I don't want to invest time in this right now

    saturation_field2_1 = np.tan(sat_cond * np.pi / 2) / result.params['g'].value + HEB2 - HC2
    saturation_field2_2 = np.tan(sat_cond * np.pi / 2) / result.params['g'].value + HEB2 + HC2
    saturation_field2_3 = np.tan(-sat_cond * np.pi / 2) / result.params['g'].value + HEB2 - HC2
    saturation_field2_4 = np.tan(-sat_cond * np.pi / 2) / result.params['g'].value + HEB2 + HC2
    saturation_fields2 = (np.min([saturation_field2_1, saturation_field2_2, saturation_field2_3, saturation_field2_4]),
                          np.max([saturation_field2_1, saturation_field2_2, saturation_field2_3, saturation_field2_4]))
    dsaturation_fields2 = (half_step_size, half_step_size) #TODO: This is only partly true, but I don't want to invest time in this right now

    integral_args = result.params.valuesdict()
    integral_args['a'] -= np.min(ydata)
    left_integral_args = {
        'a': integral_args['a'],
        'b': integral_args['b'],
        'c': integral_args['c'],
        'd': integral_args['d'],
        'e': integral_args['e'],
    }
    right_integral_args = {
        'a': integral_args['a'],
        'b': integral_args['f'],
        'c': integral_args['g'],
        'd': integral_args['h'],
        'e': integral_args['i'],
    }

    # integral of both hysteresis loops together (see tan_hys for calculation)
    integral_leftbranch, integral_leftbranch_err = quad(double_arctan, np.min(xdata), np.max(xdata), args=tuple(integral_args.values()))
    integral_args['e'] *= -1
    integral_args['i'] *= -1
    integral_rightbranch, integral_rightbranch_err = quad(double_arctan, np.min(xdata), np.max(xdata), args=tuple(integral_args.values()))
    integral = integral_leftbranch - integral_rightbranch
    integral_err = np.sqrt(integral_leftbranch_err**2 + integral_rightbranch_err**2)

    # integral of both hysteresis loops separately
    left_integral_leftbranch, left_integral_leftbranch_err = quad(arctan, np.min(xdata), np.max(xdata), args=tuple(left_integral_args.values()))
    left_integral_args['e'] *= -1
    left_integral_rightbranch, left_integral_rightbranch_err = quad(arctan, np.min(xdata), np.max(xdata), args=tuple(left_integral_args.values()))
    left_integral = left_integral_leftbranch - left_integral_rightbranch
    left_integral_err = np.sqrt(left_integral_leftbranch_err**2 + left_integral_rightbranch_err**2)

    right_integral_leftbranch, right_integral_leftbranch_err = quad(arctan, np.min(xdata), np.max(xdata), args=tuple(right_integral_args.values()))
    right_integral_args['e'] *= -1
    right_integral_rightbranch, right_integral_rightbranch_err = quad(arctan, np.min(xdata), np.max(xdata), args=tuple(right_integral_args.values()))
    right_integral = right_integral_leftbranch - right_integral_rightbranch
    right_integral_err = np.sqrt(right_integral_leftbranch_err**2 + right_integral_rightbranch_err**2)

    fitted_data = {
        'xdata': xdata, 
        'ydata': ydata,
        'xdata_err': xdata_err,
        'ydata_err': ydata_err,
    }
    params = { #integral missing right now
        'r_squared': result.rsquared, 
        'HEB1': HEB1, 
        'dHEB1': dHEB1,
        'HC1': HC1, 
        'dHC1': dHC1,
        'HEB2': HEB2, 
        'dHEB2': dHEB2,
        'HC2': HC2, 
        'dHC2': dHC2,
        'MR': MR,
        'dMR': dMR,
        'MHEB1': MHEB1,
        'dMHEB1': dMHEB1,
        'MHEB2': MHEB2,
        'dMHEB2': dMHEB2,
        'integral': integral,
        'dintegral': integral_err,
        'integral1': left_integral,
        'dintegral1': left_integral_err,
        'integral2': right_integral,
        'dintegral2': right_integral_err,
        'saturation_fields1': saturation_fields1,
        'dsaturation_fields1': dsaturation_fields1,
        'saturation_fields2': saturation_fields2,
        'dsaturation_fields2': dsaturation_fields2,

        'ratio': {
            'HEB1/HEB2': HEB1/HEB2 if HEB2 != 0 else np.inf, 
            'HC1/HC2': HC1/HC2 if HC2 != 0 else np.inf, 
            'A1/A2': result.params['b'].value/result.params['f'].value if result.params['f'].value != 0 else np.inf,
            'area1/area2': left_integral/right_integral if right_integral != 0 else np.inf,
            },
        
        'slope_atHC1': slope_atHC1,
        'dslope_atHC1': dslope_atHC1,
        'slope_atHEB1': slope_atHEB1,
        'dslope_atHEB1': dslope_atHEB1,
        'slope_atHC2': slope_atHC2,
        'dslope_atHC2': dslope_atHC2,
        'slope_atHEB2': slope_atHEB2,
        'dslope_atHEB2': dslope_atHEB2,
        'alpha1': alpha1,
        'dalpha1': dalpha1,
        'rectangularity1': rectangularity1,
        'drectangularity1': drectangularity1,
        'alpha2': alpha2,
        'dalpha2': dalpha2,
        'rectangularity2': rectangularity2,
        'drectangularity2': drectangularity2,
        
        'x_unit': x_unit,
        'y_unit': y_unit,
    }
    
    return fitted_data, params, result

#%%
###############################################################################
# 6. Further functions, mainly for conversion and plotting
###############################################################################

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

def safe_stderr(stderr):
    """
    Ensure that lmfit results without stderr do not crash the program.

    This function checks if the provided stderr is None and returns 0 in that case.
    Otherwise, it returns the provided stderr value.

    Parameters
    ----------
    stderr : float or None
        The standard error value to check. It can be a float or None.

    Returns
    -------
    float
        Returns 0 if stderr is None, otherwise returns the provided stderr value.

    Examples
    --------
    >>> safe_stderr(None)
    0
    >>> safe_stderr(0.05)
    0.05
    """
    return 0 if stderr is None else stderr

def unit_converter(series: pd.Series, conversion_factors: dict, target_unit: str) -> pd.Series:
    """
    Convert a pandas Series from one unit to another using provided conversion factors.

    This function converts the values in a pandas Series from the current unit to a target unit
    based on the provided conversion factors. The Series must have a 'unit' attribute that specifies
    the current unit of the values.

    Parameters
    ----------
    series : pd.Series
        The pandas Series to be converted. Must have a 'unit' attribute.
    conversion_factors : dict
        A dictionary containing conversion factors. The keys are quantity types (e.g., 'length', 'mass'),
        and the values are dictionaries mapping units to their conversion factors relative to a base unit.
        Example: {'length': {'m': 1, 'cm': 0.01, 'mm': 0.001}}.
    target_unit : str
        The unit to which the Series values should be converted.

    Returns
    -------
    pd.Series
        A new pandas Series with values converted to the target unit. The 'unit' attribute of the Series
        is updated to the target unit.

    Raises
    ------
    AttributeError
        If the input Series does not have a 'unit' attribute.
    ValueError
        If the conversion_factors dictionary is empty.
        If the current unit or target unit is not supported by the conversion_factors dictionary.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([100, 200, 300])
    >>> data.unit = 'cm'
    >>> conversion_factors = {'length': {'m': 1, 'cm': 0.01, 'mm': 0.001}}
    >>> unit_converter(data, conversion_factors, 'm')
    0    1.0
    1    2.0
    2    3.0
    dtype: float64
    """
    # Ensure the series has a 'unit' attribute
    if not hasattr(series, 'unit'):
        raise AttributeError("The pd.Series object must have a 'unit' attribute.")
    
    # Ensure the dictionary of conversion factors is not empty
    if not conversion_factors:
        raise ValueError("The dictionary of conversion factors cannot be empty.")

    current_unit = series.unit
    quantity_type = None

    for key in conversion_factors:
        if str(current_unit) in conversion_factors[key].keys():
            quantity_type = key
            break
        
    # Ensure the current and target units are in the conversion dictionary
    if quantity_type is None or target_unit not in conversion_factors[quantity_type]:
        all_units = {key: list(units.keys()) for key, units in conversion_factors.items()}
        raise ValueError(f"Unsupported unit. Supported units are: {all_units}")

    # Convert to base unit of quantity type
    series_in_base_unit = series * conversion_factors[quantity_type][current_unit]

    # Convert from base unit to target unit
    converted_series = series_in_base_unit / conversion_factors[quantity_type][target_unit]

    # Update the unit attribute
    converted_series.unit = target_unit

    return converted_series
