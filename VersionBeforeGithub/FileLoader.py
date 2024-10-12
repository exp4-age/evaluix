#%%
import dataclasses
import os
#import cv2
import linecache
import logging
import sys
from typing import Any, Callable
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from NSFopen import read as nid_read  # for reading in .nid files
import pathlib

# read in data
# optical MOKE hysteresis (data points and images); always apply normalization
# VSM hysteresis (only apply hysteresis if onw wishes to)

from PyQt6 import QtWidgets

# %%
"""
following functions are capable of reading in:
    1. The manually selected file(s) via get_files() function
    2. All files in a directory via get_directory() function
    3. All files of all children directories of a given directory via the
    get_directories() function

in all functions, the file format is checked 
    
"""

###########################################
# Define a logger for this module
###########################################
own_path = pathlib.Path(__file__).parent.absolute()
logger_name = __name__

# Create a logger for extensive file logging
file_logger = logging.getLogger(f"{logger_name}.FileLogger")
file_logger.setLevel(logging.DEBUG)

# Create a logger for console logging
console_logger = logging.getLogger(f"{logger_name}.ConsoleLogger")
console_logger.setLevel(logging.WARNING) #only log warnings and errors to the console

# Get the current date as a string in the format 'YYYY-MM-DD'
date_str = datetime.date.today().strftime('%Y-%m-%d')

# Create a file handler and set its level to DEBUG
fh = logging.FileHandler(own_path / f'logger_{date_str}.log')
fh.setLevel(logging.DEBUG)

# Create a console handler and set its level to INFO
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# Add the handlers to the logger
file_logger.addHandler(fh)
console_logger.addHandler(ch)

def log_message(level, message):
    if level == 'debug':
        file_logger.debug(message)
        console_logger.debug(message)
    elif level == 'info':
        file_logger.info(message)
        console_logger.info(message)
    elif level == 'warning':
        file_logger.warning(message)
        console_logger.warning(message)
    elif level == 'error':
        file_logger.error(message)
        console_logger.error(message)

#Levels
#DEBUG: Detailed information, typically of interest only when diagnosing problems.
#INFO: Confirmation that things are working as expected.
#WARNING: An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.
#ERROR: Due to a more serious problem, the software has not been able to perform some function.
#CRITICAL: A serious error, indicating that the program itself may be unable to continue running.


###########################################
# Define the data structure
###########################################

# global dictionary to store all data objects
class Data:
    # initialize the data dictionary
    def __init__(self):
        log_message('info', "Initializing the data dictionary.")
        self.dataset = {}
        self.id = 0  # initialize the internal id of the data object

    # add a data object to the dictionary
    def add_dataset(self, dataset):
        log_message('info', f"Adding dataset with id {self.id} to the data dictionary.")
        self.dataset[self.id] = dataset
        self.id += 1

    # delete a data object from the dictionary
    def del_dataset(self, id):
        log_message('info', f"Deleting dataset with id {id} from the data dictionary.")
        del self.dataset[id]
        
    def write_specific_dataset(self, id, dataset):
        log_message('info', f"Writing dataset with id {id} to the data dictionary.")
        self.dataset[id] = dataset
        # to not overwrite data afterwards, set the id above the highest id in the dictionary
        ids = list(self.dataset.keys())
        self.id = max(ids) + 1
    
    # def assign_id(self, id):
    #     logger.info(f"Assigning id {id} to the data dictionary.")
    #     self.id = id
    
    # Make the data object iterable
    def __iter__(self):
        for id in self.dataset:
            yield self.dataset[id]
            
    # information about the data object
    def info(self):
        # number of datasets in the dictionary
        n_datasets = len(self.dataset)
        # list of all dataset ids
        dataset_ids = list(self.dataset.keys())
        # list of all dataset metadata
        dataset_metadata = [self.dataset[key].metadata for key in dataset_ids]
        # total amount of bytes of all datasets
        total_size_bytes = sum([sys.getsizeof(self.dataset[key]) for key in dataset_ids])
        
        # Units of measurement for data size
        units = ["bytes", "kB", "MB", "GB", "TB", "PB"]
        unit_index = 0
        
        # Convert total size to appropriate unit
        while total_size_bytes >= 1024 and unit_index < len(units) - 1:
            total_size_bytes /= 1024.0
            unit_index += 1
        
        # Format total size with appropriate unit
        total_size_formatted = f"{total_size_bytes:.2f} {units[unit_index]}"
        
        info = {
            "n_datasets": n_datasets,
            "total_size": total_size_formatted,
            "dataset_ids": dataset_ids,
            "dataset_metadata": dataset_metadata,
        }
        
        return info
            
    # clear the data dictionary completely
    def clear(self):
        log_message('info', "Clearing the data dictionary.")
        self.dataset = {}
        self.id = 0
        
        if not self.dataset:
            log_message('info', "Data dictionary is empty.")

# initialize the global object data
#data = Data()

@dataclasses.dataclass  # dataclass decorator, automatically generates __init__ and __repr__ methods
class Dataset:
    # initialize the data properties in dataclass
    metadata = dict
    raw_data = pd.DataFrame

    def __init__(
        self,
        metadata: dict,
        raw_data: pd.DataFrame,
    ):
        log_message('debug', "Initializing the data object.")
        self.metadata = metadata
        self.raw_data = raw_data

    # representation of the data object
    def __str__(self):
        return str(self.metadata) + str(self.raw_data.info)

    def add_log_mod_results(self, key: int, loglist: list, mod_data: pd.DataFrame, results: dict = {}):
        log_message('debug', f"Adding log_mod_results with key {key} to the data object.")
        setattr(self, f"loglist_{key}", loglist)
        setattr(self, f"mod_data_{key}", mod_data)
        setattr(self, f"results_{key}", results)
    
    # delete the data_mod attribute if desired
    def del_log_mod_results(self, key: int):
        try:
            log_message('debug', f"Deleting log_mod_results with key {key} from the data object.")
            delattr(self, f"loglist_{key}")
            delattr(self, f"mod_data_{key}")
            delattr(self, f"results_{key}")
        # if the attribute does not (yet) exist, pass
        except AttributeError as e:
            log_message('warning', f"The attribute does not exist. Error: {str(e)}")
            pass
        
    def update_mod_data(self, key: int, new_data):
        # Check if new_data is a dictionary
        if isinstance(new_data, dict):
            # Convert dictionary to DataFrame
            new_data = pd.DataFrame(new_data)

        # Replace the existing DataFrame with the new one
        log_message('debug', f"Updating mod_data with key {key} in the data object.")
        setattr(self, f"mod_data_{key}", new_data)
        
    def items(self):
        for attr in vars(self):
            yield attr, getattr(self, attr)
            
    def info(self):
        return self.metadata, self.raw_data.info()

###########################################
# Define the global read_file function, which is separated into data and metadata functions
###########################################

# read in data from a file
def read_file(data: Data, file_or_dir: str, dataformat: str):
    log_message('info', f"Reading file or directory {file_or_dir} with data format {dataformat}")
    # Check if file_or_dir is a directory
    if os.path.isdir(file_or_dir):
        log_message('debug', f"{file_or_dir} is a directory. Reading all files in the directory.")
        # If it's a directory, call read_file recursively on all files in the directory
        for filename in os.listdir(file_or_dir):
            read_file(data, os.path.join(file_or_dir, filename), dataformat)
    else:
        log_message('debug', f"{file_or_dir} seems to be a file. Reading the file.")
        # If it's not a directory, proceed as before
        if file_or_dir is None:
            logger.warning("No file selected")
            return "No file selected"

        if not isinstance(data, Data):
            log_message('warning', "No correct data object found. Creating a new one.")
            data = Data()

        if format_checker(file_or_dir, dataformat):
            metadata = get_metadata(file_or_dir, dataformat, data.id)
            metadata = metadata_checker(metadata)
            raw_data = get_data(file_or_dir, dataformat, metadata)
            
            # Dictionary to store units for each column
            column_units = {}

            # Separate name and unit, rename the columns by the pure name and add the unit to the dictionary
            for col in raw_data.keys():
                if "[" in col:
                    name, unit = col.split("[")
                    raw_data.rename(columns={col: name.strip()}, inplace=True)
                    raw_data[name.strip()] = raw_data[name.strip()].astype(float)
                    column_units[name.strip()] = unit.strip()[:-1]
                elif "(" in col:
                    name, unit = col.split("(")
                    raw_data.rename(columns={col: name.strip()}, inplace=True)
                    raw_data[name.strip()] = raw_data[name.strip()].astype(float)
                    column_units[name.strip()] = unit.strip()[:-1]

            # Reapply units to columns. For unknown reason, applying the units in the previous loop does not work
            for col, unit in column_units.items():
                raw_data[col].unit = unit
            
            data.add_dataset(Dataset(metadata, raw_data))
            log_message('debug', f"Data successfully read from {file_or_dir}")
        
        elif not format_checker(file_or_dir, dataformat):
            log_message('error', f"The file format of {file_or_dir} does not fit the chosen data format {dataformat}. Please choose a different file or data format.")
            return f"The file format of {file_or_dir} does not fit the chosen data format {dataformat}. Please choose a different file or data format."

def get_line(file, line_number):
    with open(file, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f, start=1):
            if i == line_number:
                return line

def format_checker(file: str, dataformat: str):
    # possible file extensions for each data format
    formats = {
        "MOKE": [".txt", ".asc", ".csv", "hdf5"],
        "VSM": [".txt", ".csv", ".VSM-Hys-Data", ".VSM-HYS-DATA", "hdf5"],
        "Kerr": [".txt", ".asc", ".csv", "hdf5"],
        "Kerr_imgs": [".png", ".tiff", ".jpg", "hdf5"],
        "AFM": [".nid", ".gwy", "hdf5"],
        "SRIM": [".txt", "hdf5"],
        "MUMAX3": [".txt", ".csv", "hdf5"],
    }

    # check if file format is in the list of possible file extensions (or a directory for Kerr_imgs)
    if os.path.isdir(file) or os.path.splitext(file)[1] in [".png", ".tiff", ".jpg"]:
        log_message('debug', f"{file} is a directory or an image file.")
        if dataformat == "Kerr" or dataformat == "Kerr_imgs":
            dataformat = "Kerr_imgs"  # switch to Kerr_imgs if it's a directory or an image file
            log_message('debug', f"Data format changed to {dataformat} for directory or image file.")
            return dataformat  # directories are valid for Kerr_imgs
        else:
            log_message('error', f"{file} seems to be a directory or an image file. Please choose a different data format.")
            return False  # directories are not valid for other formats
    else:
        file_extension = os.path.splitext(file)[1]
        return file_extension in formats[dataformat]

###########################################
# Define how the program reads the metadata. Every dataformat/device has its own function
###########################################

# Thoughts on the metadata:

"""
data to be stored in the metadata dictionary. Set to False or unknown if not available:
##############################################
crucial metadata: necessary for Evaluix to treat the data correctly and to unambiguously identify the sample and measurement
    - internal_id: id of the data object in the global Data dictionary
    - uuid: unique id of the measurement/simulation
    - type: measurement type (e.g. Hyst, Img (AFM or Kerr), ImgStack, Sim, ...)
        - special case: if type is set to unknown everything of Evaluix will be enabled but take care to perform meaningful analysis/functions
    - sample: sample name and/or uuid
    - device: measurement device (e.g. VSM, Kerr, ...)
    - datetime: date and time of the measurement
    - file: file name
##############################################
for the case of a measurement series (e.g. raster L-MOKE, VSM/V-MOKE angular resolved hysteresis measurements)
    - measurement_series: True (as a list filled with the following information) or False
        - measurement_series_id: second internal id of the measurement series (e.g. 1, 2, 3, ...)
        - measurement_series_total: total number of measurements in the series n_max, 2d array for 2d measurements [x_max,y_max]
        - measurement_series_position: position of the measurement in the series n, 2d array for 2d measurements [x,y] #maybe 3d for 3d field measurement setup [x,y,z]
        - measurement_series_range: physical range of the measurement series (e.g. from 90° to 450° or from [0,0] mm to [14,10] mm)
            - TODO: if stepsize is not constant, an additional measurement_series_stepsize is necessary, discuss with CJ and Yahya
##############################################
optional metadata: not necessary for Evaluix to treat the data correctly but useful (or important) for further analysis
    - user: user who performed the measurement or simulation
    - mode: measurement mode (contact, non-contact, MFM, ...)
    - sample_status: status of the sample (e.g. as made, IB, FC, structured, ...) #maybe better as a list accumulting all the information
    - sample_angle: angle orientation of the sample (only for V-MOKE and VSM measurements)
    - field_angle: angle orientation of the applied field (only for V-MOKE, VSM and Kerr measurements)
    - temperature: temperature during the measurement; default is room temperature (20°C)
    - resolution: resolution of the measurement (only for raster L-MOKE and AFM measurements, possibly also for Kerr images)
    - calibration: calibration file name for the measurement device (e.g. VSM calibration, Kerr calibration, ...)
    - averaging: number of averages for the measurement (Hysteresis loops for VSM/L-MOKE/V-MOKE measurements, images for Kerr microscopy)
    - polarization: polarization of the light (only for Kerr measurements) or polarization of the MFM tip (only for MFM measurements)
    - sample_orientation: orientation of the sample (only for VSM measurements)
    - field_orientation: orientation of the applied field (only for Kerr microscopy)
    - time_per_point: time per point/image of the measurement (for VSM/LMOKE/VMOKE/AFM(indirect)/Kerr measurements)
    - delay_time: delay time between the measurement points (for VSM/LMOKE/VMOKE/Kerr measurements)
    - comment: comment on the measurement
"""

# This structure is tree like
    # get metadata function
        # assigns the correct function to the dataformat
        # assigned get metadata function
            # initializes a few default values
            # searches through the filename
            # searches through the file for the metadata to find the metadata or updates the default values
            # assignes process functions according to the lines in the file
                # slices the line(s) format specifically to get the metadata
            
        # returns the metadata dictionary
        
def get_metadata(file: str, dataformat: str, Data_id: int): 
    # assign the correct function to the dataformat
    dataformat_to_function = {
        "VSM": get_VSM_metadata,
        "AFM": get_AFM_metadata,
        "MOKE": get_MOKE_metadata,
        "Kerr": get_Kerr_metadata,
        "Kerr_imgs": get_Kerr_imgs_metadata,
    }

    # check if the dataformat is supported and execute the corresponding function
    if dataformat in dataformat_to_function:
        log_message('debug', f"Getting metadata for {file} with data format {dataformat}.")
        return dataformat_to_function[dataformat](file, Data_id)
    else:
        log_message('error', f"Unknown data format: {dataformat}")
        raise ValueError(f"Unknown data format: {dataformat}")

def get_device_from_filename(filename):
    if re.search(r'V[-_]MOKE|VMOKE', filename, re.IGNORECASE):
        return 'V-MOKE'
    elif re.search(r'L[-_]MOKE|LMOKE', filename, re.IGNORECASE):
        return 'L-MOKE'
    elif filename.endswith('.nid') or filename.endswith('.gwy'):
        return 'AFM'
    elif filename.endswith('.VSM-Hys-Data'):
        return 'VSM'
    else:
        return None

def process_single_line(line):
    # This function splits the given line by the last occurence of a delimiter and returns the value
    # tab for LMOKE and VMOKE
    # colon for VSM
    # space has not yet a device specific use and is used as a fallback
    if '\t' in line:
        value = line.split('\t')[-1]
    elif ':' in line:
        value = line.split(':')[-1]
    else:
        value = line.split()[-1]
    return value.strip()
    
def process_multi_line(line):
    #TODO: Comments may be over multiple lines, implement a function to process them
    if '\t' in line:
        comment = line.split('\t')[-1]
    elif ':' in line:
        comment = line.split(':')[-1]
    else:
        comment = line.split()[-1]
    return False if comment.strip() == "None" else comment.strip()

def get_MOKE_metadata(file, Data_id):
    # initialize the metadata dictionary
    metadata = {}
    metadata['internal_id'] = Data_id
    
    # first, get metadata from the filename and assign default values
    metadata['path'] = metadata.get('path', pathlib.Path(file).parent.absolute())
    metadata['file'] = metadata.get('file', os.path.basename(file))
    metadata['device'] = metadata.get('device', get_device_from_filename(file))
    metadata['type'] = metadata.get('type', 'Hyst')
    
    # second, get metadata from the file and map it to the corresponding metadata key
    line_start_to_function = {
        #LMOKE:
        "# User": ['user', process_single_line],
        "# Sample": ['sample', process_single_line],
        "# State": ['state', process_single_line],
        "# Comment": ['comment', process_multi_line],
        "# Meas. type": ['type', process_single_line],
        "# angle (deg)": ['sample_angle', process_single_line],
        "# ID": ['uuid', process_single_line],
        
        #VMOKE:
        "User": ['user', process_single_line],
        "Charge": ['sample', process_single_line],
        "Sample": ['sample', process_single_line],
        "Sample State": ['state', process_single_line],
        "Comment": ['comment', process_multi_line],
        "Measurement Method": ['type', process_single_line],
        "Sample Angle": ['sample_angle', process_single_line],
        "Magnetic Field Angle": ['field_angle', process_single_line],
        "ID": ['uuid', process_single_line],
    }

    with open(file, "r", encoding='utf-8', errors='replace') as f: # open the file
        for line in f.readlines():
            for line_start, [metadata_key, function] in line_start_to_function.items():
                if line.startswith(line_start):
                    metadata[metadata_key] = function(line)
        
    # third, change the mapped values if necessary
    if metadata['type'] == 'Raster':
        metadata['type'] = 'Hyst'
        metadata['raster'] = True
        metadata['raster_resolution'] = metadata.get('raster_resolution', False) #TODO: get the resolution from the file
    elif metadata['type'] == 'FORC' or metadata['type'] == 'Minor loop':
        metadata['type'] = metadata['type']
    else:
        metadata['type'] = 'Hyst'

    metadata['datetime'] = datetime.datetime.fromtimestamp(
        os.path.getmtime(file)
    ).strftime("%Y-%m-%d %H:%M:%S") 
    
    log_message('debug', f"Metadata for {file} successfully read.")
    return metadata

def get_VSM_metadata(file, Data_id):
    # initialize the metadata dictionary
    metadata = {}
    metadata['internal_id'] = Data_id
    
    # first, get metadata from the filename and assign default values
    metadata['device'] = metadata.get('device', get_device_from_filename(file))
    metadata['path'] = metadata.get('path', pathlib.Path(file).parent.absolute())
    metadata['file'] = metadata.get('file', os.path.basename(file))
    metadata['type'] = metadata.get('type', 'Hyst')
    
    # second, get metadata from the file and map it to the corresponding metadata key
    line_start_to_function = {
        #LMOKE:
        "@Operator:": ['user', process_single_line],
        "@Samplename:": ['sample', process_single_line],
        "@@Comments": ['comment', process_multi_line],
        "@Measurement type:": ['type', process_single_line],
    }

    with open(file, "r", encoding='utf-8', errors='replace') as f: # open the file
        for line in f.readlines():
            for line_start, [metadata_key, function] in line_start_to_function.items():
                if line.startswith(line_start):
                    metadata[metadata_key] = function(line)
                elif line.startswith("@@END Parameters"):
                    break
                
    # third, change the mapped values if necessary
    if metadata['type'] == "Hysteresis Loop": #right now only hysteresis loops are supported
        metadata['type'] = 'Hyst'
    else:
        metadata['type'] = 'unknown'

    metadata['datetime'] = datetime.datetime.fromtimestamp(
        os.path.getmtime(file)
    ).strftime("%Y-%m-%d %H:%M:%S") 
    
    log_message('debug', f"Metadata for {file} successfully read.")
    return metadata

def get_AFM_metadata(file, Data_id):
    # initialize the metadata dictionary, TODO: put more effort into the extraction of the AFM metadata
    metadata = {}
    metadata['internal_id'] = Data_id
    
    # first, get metadata from the filename and assign default values
    metadata['device'] = metadata.get('device', get_device_from_filename(file))
    metadata['path'] = metadata.get('path', pathlib.Path(file).parent.absolute())
    metadata['file'] = metadata.get('file', os.path.basename(file))
    # search for the sample name in the filename using regular expressions
    sample_name = re.search(r'\d{4}_\d{4}(_\d{1,3})?', file)
    
    # check if a match is found
    if sample_name:
        metadata['sample'] = sample_name.group()
    else:
        metadata['sample'] = "Unknown"
    
    metadata['type'] = metadata.get('type', 'Img')
    metadata['datetime'] = datetime.datetime.fromtimestamp(
        os.path.getmtime(file)
    ).strftime("%Y-%m-%d %H:%M:%S")
    
    log_message('debug', f"Metadata for {file} successfully read.")
    return metadata

def get_Kerr_metadata(file, Data_id):
    # initialize the metadata dictionary
    metadata = {}
    metadata['internal_id'] = Data_id
    
    # search for the sample name in the filename using regular expressions
    sample_name = re.search(r'\d{4}_\d{4}(_\d{1,3})?', file)
    
    # check if a match is found
    if sample_name:
        metadata['sample'] = sample_name.group()
    else:
        metadata['sample'] = "Unknown"
    
    metadata['device'] = metadata.get('device', 'Kerr')
    metadata['path'] = metadata.get('path', pathlib.Path(file).parent.absolute())
    metadata['file'] = metadata.get('file', os.path.basename(file))
    metadata['type'] = metadata.get('type', 'Hyst')
    metadata['datetime'] = datetime.datetime.fromtimestamp(
        os.path.getmtime(file)
    ).strftime("%Y-%m-%d %H:%M:%S")
    
    log_message('debug', f"Metadata for {file} successfully read.")
    return metadata

def get_Kerr_imgs_metadata(file, Data_id):
    # initialize the metadata dictionary
    metadata = {}
    metadata['internal_id'] = Data_id
    
    # search for the sample name in the filename using regular expressions
    sample_name = re.search(r'\d{4}_\d{4}(_\d{1,3})?', file)
    
    # check if a match is found
    if sample_name:
        metadata['sample'] = sample_name.group()
    else:
        metadata['sample'] = "Unknown"
    
    metadata['device'] = metadata.get('device', 'Kerr')
    metadata['path'] = metadata.get('path', pathlib.Path(file).parent.absolute())
    metadata['file'] = metadata.get('file', os.path.basename(file))
    metadata['type'] = metadata.get('type', 'Img')
    metadata['datetime'] = datetime.datetime.fromtimestamp(
        os.path.getmtime(file)
    ).strftime("%Y-%m-%d %H:%M:%S")
    
    if os.path.isdir(file):
        img_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']
        metadata['Nr_of_imgs'] = len([f for f in os.listdir(file) if os.path.splitext(f)[1] in img_extensions])
    
    log_message('debug', f"Metadata for {file} successfully read.")
    return metadata

def metadata_checker(metadata: dict):
    # check if the crucial parts of metadata are complete
    # if all([metadata["type"], metadata["sample"], metadata["device"]]):
    #     return metadata  # if yes, return the metadata dictionary
    # # if not, create a dialog window to input the missing metadata
    # else:
    #     print(metadata)
    #     return None
    #     # input_metadata(metadata)
    
    # TODO: redo uncommented part
    return metadata

def datalocator(file: str, fileformat: str):
    # working file formats: VSM
    data_start = None
    data_end = None
    with open(file, "r", encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
        if fileformat == "VSM":
            for line in lines:
                if re.search("New Section: Section 0: \n", line):
                    data_start = lines.index(line) + 1
                    break
            for line in lines[data_start:]:
                if re.search("@@END Data.\n", line):
                    data_end = len(lines) - lines.index(line)
                    break
                    
        if fileformat == "MOKE":
            for line in lines:
                if re.search("^# -+", line):
                    data_start = lines.index(line) + 2
                    data_end = 0
                    break
                
        if fileformat == "Kerr": # for Hystereses with the commercial software
            for line in lines:
                if re.search("^Field", line):
                    data_start = lines.index(line) + 1
                    data_end = 0
                    break
                
    # if not data_start or not data_end:
    #     log_message('error', f"Data start and/or end not found in {file}.")
        
    return data_start, data_end

def get_data(file: str, dataformat: str, metadata: dict = {}):
    log_message('debug', f"Getting data from {file} with data format {dataformat}.")
    # read in the data
    if dataformat == "AFM":
        log_message('debug', f"Reading AFM data from {file}.")
        _data = nid_read(file)  # read in the data
        _data = {
            "forward": list(1e9 * np.flip(_data.data[1], axis=0)),
            "backward": list(1e9 * np.flip(_data.data[3], axis=0)),
        }  # extract the forward and backward data

        return pd.DataFrame(_data)  # return the data as a pandas dataframe

    if dataformat == "VSM":
        log_message('debug', f"Reading VSM data from {file}.")
        # get the line numbers in which the data starts and ends
        data_start, data_end = datalocator(file, dataformat)
        # check if other measurement modi have other data types
        col_names = [
            "Time [s]", # column 0
            "TemperatureRaw [degC]", # column 1
            "Temperature1 [degC]", # column 2
            "Temperature2 [degC]", # column 3
            "Applied FieldRaw [Oe]", # column 4
            "Applied Field [Oe]", # column 5
            "Field Angle [deg]", # column 6
            "Applied FieldRawPlot [Oe]", # column 7
            "Applied FieldPlot [Oe]", # column 8
            "MomentMxRaw [memu]", # column 9
            "MomentMyRaw [memu]", # column 10
            "MomentMx [emu]", # column 11
            "MomentMy [emu]", # column 12
        ]

        with open(file, 'r', encoding='utf-8', errors='replace') as f:
            df = pd.read_csv(
                f,
                names=col_names,
                skiprows=data_start,  # this counts logically from top to bottom
                skipfooter=data_end,  # this stupid mode starts to count from the BOTTOM to the top
                sep="   ",
                engine="python",
            )
            
        # update field name to "H (Oe)" and moment name to "M (emu)"
        log_message('debug', f"Renaming columns for VSM data from {file}.")
        df.rename(
            columns={
                "Applied FieldRawPlot [Oe]": "H (Oe)",
                "MomentMxRaw [memu]": "Mx (memu)",
                "MomentMyRaw [memu]": "My (memu)",
            },
            inplace=True,
        )
        if len(df) % 2 != 0:
            log_message('debug', f"Data in {file} is not even. Adding a row to make it even.")
            # get the index of the center row
            center_index = len(df) // 2
            # insert a row after the center row with the same values
            df = pd.concat(
                [df[:center_index+1], # first half of the data
                    df.loc[center_index: center_index].copy(), # added row
                    df[center_index+1:]] # second half of the data
                ).reset_index(drop=True) # reset the index of the dataframe

        # check if y data is present and return the data accordingly
        if np.sum(np.abs(df["My (memu)"])) > 0:
            return df[["H (Oe)", "Mx (memu)", "My (memu)"]]
        else:
            return df[["H (Oe)", "Mx (memu)"]]
        
    if dataformat == "MOKE":
        log_message('debug', f"Reading MOKE data from {file}.")
        # get the line numbers in which the data starts and ends
        data_start, data_end = datalocator(file, dataformat)
        
        # extract the column names from the file (one above the data start)
        line = get_line(file, data_start)
                
        if line.startswith("#"):
            line = line.split('#')[-1]
            
        col_names = [col_name.strip() for col_name in line.split("\t")] # split the line by tabs and assign the values to the column names
        # print(data_start, data_end, line, col_names)
        with open(file, 'r', encoding='utf-8', errors='replace') as f:
            df = pd.read_csv(
                f,
                names=col_names,
                skiprows=data_start,  # this counts logically from top to bottom
                skipfooter=data_end,  # this stupid mode starts to count from the BOTTOM to the top
                sep=r"\s+",
                engine="python",
            )
        
        # if the file contains a hysteresis loop, the data should be even so that the forward and backward data can be separated
        if metadata.get('type', '') == 'Hyst':
            if len(df) % 2 != 0:
                log_message('debug', f"Data in {file} is not even. Adding a row to make it even.")
                # get the index of the center row
                center_index = len(df) // 2
                # insert a row after the center row with the same values
                df = pd.concat(
                    [df[:center_index+1], # first half of the data
                     df.loc[center_index: center_index].copy(), # added row
                     df[center_index+1:]] # second half of the data
                    ).reset_index(drop=True) # reset the index of the dataframe
        
        return df
    
    if dataformat == "Kerr":
        log_message('debug', f"Reading Kerr data from {file}.")
        # get the line numbers in which the data starts and ends
        data_start, data_end = datalocator(file, dataformat)
        
        # extract the column names from the file (one above the data start)
        line = get_line(file, data_start)
        col_names = [col_name.strip() for col_name in line.split("\t")]
        
        with open(file, 'r', encoding='utf-8', errors='replace') as f:
            df = pd.read_csv(
                f,
                names=col_names,
                skiprows=data_start,  # this counts logically from top to bottom
                skipfooter=data_end,  # this stupid mode starts to count from the BOTTOM to the top
                sep=r"\s+",
                engine="python",
            )
        
        if "File" in df.keys(): #TODO: check if the images cannot be read in as well
            df.drop(columns=["File"], inplace=True)
        
        # if the file contains a hysteresis loop, the data should be even so that the forward and backward data can be separated
        if metadata.get('type', '') == 'Hyst':
            if len(df) % 2 != 0:
                log_message('debug', f"Data in {file} is not even. Adding a row to make it even.")
                # get the index of the center row
                center_index = len(df) // 2
                # insert a row after the center row with the same values
                df = pd.concat(
                    [df[:center_index+1], # first half of the data
                     df.loc[center_index: center_index].copy(), # added row
                     df[center_index+1:]] # second half of the data
                    ).reset_index(drop=True) # reset the index of the dataframe
        
        # rename the columns to the standard names
        if "GrayLevel" in df.keys():
            df.rename(columns={"GrayLevel": "I (arb.u.)"}, inplace=True)
        if "Field(mT)" in df.keys():
            df.rename(columns={"Field": "H (mT)"}, inplace=True)
        
        return df
 
    if dataformat == "Kerr_imgs":
        log_message('debug', f"Reading Kerr image data from {file}.")
        img_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']
        
        if os.path.isdir(file):
            img_files = [f for f in os.listdir(file) if os.path.splitext(f)[1] in img_extensions]
            img_files.sort()
            img_files = [os.path.join(file, f) for f in img_files]
            
        else:
            img_files = [file]
        
        # Initialize lists to store image data and mean values
        img_data = []
        img_means = []
        img_shape = []
        img_name = []
        
        # Loop through image files
        for img_file in img_files:
            # Open image file and convert to grayscale
            img = plt.imread(img_file)
            img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]) # convert to grayscale, TODO: uzse skimage instead
            # Flatten image
            img_flattened = img.flatten(order='C')
            # Append flattened image data and mean value to lists
            img_data.append(img_flattened.tolist())
            img_means.append(img_flattened.mean())
            img_shape.append(img.shape)
            img_name.append(pathlib.Path(img_file).name)
        
        # Create DataFrame
        df = pd.DataFrame({
            'FlattendImage': img_data,
            'Mean (Img)': img_means,
            'Shape (px)': img_shape,
            'Image name': img_name,
        })
        
        return df

# %%
