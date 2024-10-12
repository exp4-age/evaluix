#%%
import pandas as pd

def unit_converter(series: pd.Series, target_unit: str) -> pd.Series:
    # Dictionary of conversion factors to and from millitesla (mT)
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

    # Ensure the series has a 'unit' attribute
    if not hasattr(series, 'unit'):
        raise AttributeError("The pd.Series object must have a 'unit' attribute.")

    current_unit = series.unit
    quantiy_type = None

    for key in conversion_factors:
        if current_unit in conversion_factors[key]:
            quantiy_type = key

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

# Example usage
data = pd.Series([1, 2, 3], name='Magnetic Field')
data.unit = 'T'
converted_data = convert_magnetic_field(data, 'mT')
print(converted_data)
print(f"New unit: {converted_data.unit}")
# %%
