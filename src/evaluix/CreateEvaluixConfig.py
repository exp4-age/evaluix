# %%
import inspect
import ast
import yaml
import EvaluationFunctions
import pathlib

own_path = pathlib.Path(__file__).parent.absolute()

#extract the version number from "__version__.py"
with open(own_path / '__version__.py', 'r') as file:
    version = file.read().split('=')[1].strip().strip("'")

AboutEvaluix = {
    'Version': version,
    'EvaluixConfig': 'EvaluixConfig.yaml',
    'ProfileConfig': 'DefaultProfile.yaml',
    'Autosave_status': False,
}

# Default macros for the different evaluation types
DefaultMacros = {
    "Hysteresis": {
        "MOKE single loop": {
            1: {
                "Active": True,
                "Category": "Data Manipulation",
                "Function": "del_outliers",
                "Parameters": {
                    "ydata": "-",
                    "neighbours": 5,
                    "threshold": 2
                }
            },
            2: {
                "Active": True,
                "Category": "Data Manipulation",
                "Function": "rmv_opening",
                "Parameters": {
                    "ydata": "-",
                    "sat_region": 0.05
                }
            },
            3: {
                "Active": True,
                "Category": "Data Manipulation",
                "Function": "slope_correction",
                "Parameters": {
                    "xdata": "-",
                    "ydata": "-",
                    "sat_region": 0.1,
                    "noise_threshold": 3,
                    "branch_difference": 0.3
                }
            },
            4: {
                "Active": True,
                "Category": "Data Manipulation",
                "Function": "hys_norm",
                "Parameters": {
                    "xdata": "-",
                    "ydata": "-",
                    "sat_region": 0.1
                }
            },
            5: {
                "Active": True,
                "Category": "Data Evaluation",
                "Function": "tan_hyseval",
                "Parameters": {
                    "xdata": "-",
                    "ydata": "-",
                    "sat_cond": 0.95
                }
            }
        },
        "VSM single loop": {
            1: {
                "Active": True,
                "Category": "Data Manipulation",
                "Function": "del_outliers",
                "Parameters": {
                    "ydata": "-",
                    "neighbours": 5,
                    "threshold": 2
                }
            },
            2: {
                "Active": True,
                "Category": "Data Manipulation",
                "Function": "rmv_opening",
                "Parameters": {
                    "ydata": "-",
                    "sat_region": 0.05
                }
            },
            3: {
                "Active": True,
                "Category": "Data Manipulation",
                "Function": "slope_correction",
                "Parameters": {
                    "xdata": "-",
                    "ydata": "-",
                    "sat_region": 0.1,
                    "noise_threshold": 3,
                    "branch_difference": 0.3
                }
            },
            4: {
                "Active": True,
                "Category": "Data Manipulation",
                "Function": "hys_center",
                "Parameters": {
                    "xdata": "-",
                    "ydata": "-",
                    "sat_region": 0.1,
                    "normalize": False
                }
            },
            5: {
                "Active": True,
                "Category": "Data Evaluation",
                "Function": "tan_hyseval",
                "Parameters": {
                    "xdata": "-",
                    "ydata": "-",
                    "sat_cond": 0.95
                }
            }
        }
    }
}

def get_function_info(func):
    source = inspect.getsource(func)
    tree = ast.parse(source)

    func_def = tree.body[0]
    args = func_def.args

    info = {}
    defaults = args.defaults if args.defaults else []
    defaults = [None] * (len(args.args) - len(defaults)) + defaults

    for arg, default in zip(args.args, defaults):
        name = arg.arg
        annotation = ast.get_source_segment(source, arg.annotation) if arg.annotation else '-'
        default = ast.get_source_segment(source, default) if default else '-'
        
        # Remove surrounding quotes from the default value if present
        if isinstance(default, str):
            default = default.strip("'\"")
            
        info[name] = {'type': annotation, 'value': default, 'default': default}
        
    # Extract additional information from the docstring
    docstring = inspect.getdoc(func)
    if docstring:
        for line in docstring.splitlines():
            line = line.strip()
            if line.startswith("Category:"):
                info['Category'] = line.split(":")[1].strip()
                
    # Check if Category is present in the info, if not, add "unknown"
    if 'Category' not in info:
        info['Category'] = 'unknown'

    return info

# Get all functions in the module
all_functions = inspect.getmembers(EvaluationFunctions, inspect.isfunction)

# Filter out imported functions
module_functions = [func for name, func in all_functions if func.__module__ == EvaluationFunctions.__name__]

function_info = {func.__name__: get_function_info(func) for func in module_functions}

# Dictionary of conversion factors for different quantities
conversion_factors = {
    'conversion_factors' : {
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
}

own_path = pathlib.Path(__file__).parent.absolute()

# Export to EvaluixConfig.yaml
with open(own_path / 'EvaluixConfig.yaml', 'w') as file:
    yaml.dump({'Version': AboutEvaluix['Version']}, file)
    yaml.dump({'EvaluixConfig': AboutEvaluix['EvaluixConfig']}, file)
    yaml.dump({'ProfileConfig': AboutEvaluix['ProfileConfig']}, file)
    yaml.dump(conversion_factors, file)

# Export to DefaultProfile.yaml
with open(own_path / 'DefaultProfile.yaml', 'w') as file:
    yaml.dump({'Autosave_status': AboutEvaluix['Autosave_status']}, file)
    yaml.dump({'Macros': DefaultMacros}, file)
    yaml.dump({'function_info': function_info}, file)
# %%