import numpy as np
import pandas as pd
import warnings

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors

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
    titel=None, 
    color=None,
    vmin=0,
    vmax=1,):
    if ax is None:
        ax = plt.gca()

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
    if isinstance(color, str):
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
        ax.fill(polygon_x_inc, polygon_y_inc, color=color_inc, alpha=0.3, zorder=2)
        ax.fill(polygon_x_dec, polygon_y_dec, color=color_dec, alpha=0.3, zorder=2)
        
    if isinstance(color, colors.Colormap):
        # Map color values for both branches using ScalarMappable for proper colorbar support
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=color)
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
                _label = label if plotstyle == 'line' else None  # Only set label for the line plot to avoid duplicate legend entries
                lc_inc = colored_line(xdata_inc, ydata_inc, c_inc, ax, cmap=color, linewidth=2, label=_label)
                lc_inc.set_norm(norm)
            if len(xdata_dec) > 0:
                lc_dec = colored_line(xdata_dec, ydata_dec, c_dec, ax, cmap=color, linewidth=2)
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
                ax.scatter(xdata_inc, ydata_inc, c=c_inc, cmap=color, norm=norm, label=label, zorder=3, s=20)
            if len(xdata_dec) > 0:
                ax.scatter(xdata_dec, ydata_dec, c=c_dec, cmap=color, norm=norm, zorder=3, s=20)
        
    # plot the data points for both branches
    elif plotstyle == 'line' or plotstyle == 'both':
        if len(xdata_inc) > 0:
            ax.plot(xdata_inc, ydata_inc, label=label, color=color_inc, zorder=3)
        if len(xdata_dec) > 0:
            ax.plot(xdata_dec, ydata_dec, color=color_dec, zorder=3)
        if plotstyle == 'both':
            if len(xdata_inc) > 0:
                ax.scatter(xdata_inc, ydata_inc, color=color_inc, zorder=3, s=20, alpha=0.6)
            if len(xdata_dec) > 0:
                ax.scatter(xdata_dec, ydata_dec, color=color_dec, zorder=3, s=20, alpha=0.6)
    elif plotstyle == 'scatter':
        if len(xdata_inc) > 0:
            ax.scatter(xdata_inc, ydata_inc, label=label, color=color_inc, zorder=3, s=20)
        if len(xdata_dec) > 0:
            ax.scatter(xdata_dec, ydata_dec, color=color_dec, zorder=3, s=20)
    else:
        raise ValueError("Invalid plotstyle. Use 'line', 'scatter' or 'both' or change color to a cmap.")

    # style the plot
    ax.grid(ls='--', alpha=0.5, zorder=0)
    # check if the data crosses the absissa or ordinate and only plot the corresponding axis if it does
    if np.any(x < 0) and np.any(x > 0):
        ax.axvline(0, color='k', linestyle='--', zorder=1)
    if np.any(y < 0) and np.any(y > 0):
        ax.axhline(0, color='k', linestyle='--', zorder=1)
    ax.tick_params(direction='in', top=True, right=True)
    ax.set_xlabel(f'H ({xunit})' if xunit is not None else 'H (arb. u.)')
    ax.set_ylabel(f'M ({yunit})' if yunit is not None else 'M (arb. u.)')
    if titel is not None:
        ax.set_title(titel)
    ax.legend()
    ax.grid(True)
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

def convert_units(df, unit_dict):
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
        for _key in conversion_factors:
            if value in conversion_factors[_key]:
                quantity_type = _key
                break
        if quantity_type is None:
            raise ValueError(f"Unit {value} is not recognized.")
        
        # Convert the column to the base unit of the quantity type, i.e. multiply by the conversion factor
        df[key] = df[key] * conversion_factors[quantity_type][df.attrs[key]]
        
        # Now convert it to the final unit using the reciprocal of the conversion factor
        df[key] = df[key] / conversion_factors[quantity_type][value]
    
    return df

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
