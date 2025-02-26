import numpy as np
import pandas as pd

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