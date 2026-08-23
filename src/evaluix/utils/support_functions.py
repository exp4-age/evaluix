import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.collections import LineCollection

def _convert_units(df, unit_dict):
    """
    Convert units of a dataframe using a dictionary of conversion factors.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with columns to be converted.
    unit_dict : dict
        Dictionary of final units for each column in the dataframe.
        The dictionary should have the following structure:
        {
            'column_name1': 'final_unit1',
            'column_name2': 'final_unit2',
            ...
            e.g. 'H': 'mT',
        }
        
    Returns
    -------
    pandas.DataFrame
        Dataframe with converted units.
        
    """
    # Dictionary of conversion factors for different quantities
    conversion_factors = {
        'magnetic_field': {
            'mT': 1,            # millitesla to millitesla, millitesla is the base unit of this function
            'uT': 0.001,        # microtesla to millitesla
            'T': 1000,          # Tesla to millitesla
            'mG': 0.0001,       # milligauss to millitesla
            'G': 0.1,           # Gauss to millitesla
            'kG': 100,          # kilogauss to millitesla
            'mOe': 0.0001,      # millioersted to millitesla
            'Oe': 0.1,          # Oersted to millitesla (assuming 1 Oe = 1 G in vacuum)
            'kOe': 100,         # kilooersted to millitesla
            'A/m': 0.00125664,  # ampere per meter to millitesla
            'kA/m': 1.25664,    # kiloampere per meter to millitesla
            'kA/cm': 125.664,   # kiloampere per centimeter to millitesla
            'A/cm': 0.125664,   # ampere per centimeter to millitesla
        },
        'magnetic_moment': {
            'emu': 1,           # electromagnetic unit to electromagnetic unit, electromagnetic unit is the base unit of this function
            'memu': 1000,       # milli-electromagnetic unit to electromagnetic unit
            'Am2': 1000,        # ampere meter squared to electromagnetic unit
            'J/T': 1000,        # joule per tesla to electromagnetic unit
            'erg/G': 1,         # erg per gauss to electromagnetic unit
            'J/mT': 1,          # joule per millitesla to electromagnetic unit
        },
        'length': {
            'm': 1,             # meter to meter, meter is the base unit of this function
            'cm': 0.01,         # centimeter to meter
            'mm': 0.001,        # millimeter to meter
            'um': 0.000001,     # micrometer to meter
            'nm': 1e-9,         # nanometer to meter
            'pm': 1e-12,        # picometer to meter
        },
    }
    
    # Get the current units of the columns named in unit_dict
    # The units are stored as dictionary in df.attr
    for key, value in unit_dict.items():
        #check if key is column name in df and a key in df.attr
        if key not in df.columns or key not in df.attrs:
            raise ValueError(f"Unit of column {key} cannot be converted because it is not in the dataframe and/or does not have a unit attribute.")
        #check if value is a key in conversion_factors
        quantity_type = None
        for _key, _value in conversion_factors.items():
            if value in _value:
                quantity_type = _key
                break
        if quantity_type is None:
            raise ValueError(f"Unit {value} is not recognized.")
        
        # Convert the column to the base unit of the quantity type, i.e. multiply by the conversion factor
        df[key] = df[key] * conversion_factors[quantity_type][df.attrs[key]]
        
        # Now convert it to the final unit using the reciprocal of the conversion factor
        df[key] = df[key] / conversion_factors[quantity_type][value]
    
    return df

def _safe_stderr(stderr):
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

def _check_array_like(value, name):
      if not isinstance(value, (int, float, list, pd.DataFrame, pd.Series, np.ndarray)):
          raise TypeError(f'{name} must be a pandas dataframe/series, list, numpy array or int/float, not {type(value)}')

def _create_uncertainty_polygon(xdata: pd.DataFrame, ydata: pd.DataFrame, xdata_err: pd.DataFrame, ydata_err: pd.DataFrame):
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
