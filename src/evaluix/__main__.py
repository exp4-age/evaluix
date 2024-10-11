from PyQt6 import uic, QtCore
from PyQt6.QtGui import QStandardItemModel, QAction, QStandardItem, QDesktopServices
# from PyQt6.QSci import QsciScintilla, QsciLexerPython #TODO: Check if this is going to be supported in PyQt6
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QLabel,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from PyQt6.QtCore import Qt, QCoreApplication, QUrl, QItemSelection, QDir
import inspect # to get the names and docstrings of the functions
import numpy as np
import pandas as pd
import copy
import sys
import datetime
import yaml
import traceback
import h5py
from typing import Union, get_args
import pathlib

#paths
own_path = pathlib.Path(__file__).parent.absolute()
ui_path = own_path / "GUIs/Evaluix2_MainWindowLayout.ui"
created_config_path = own_path / "CreateEvaluixConfig.py"

QDir.addSearchPath('icons', str(own_path / 'Icons'))

with open(own_path / "CreateEvaluixConfig.py") as file:
    exec(file.read())
def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)
EvaluixConfig = load_config(own_path / 'EvaluixConfig.yaml')
ProfileConfig = load_config(own_path / 'DefaultProfile.yaml')

def str_to_bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError(f"Cannot convert {s} to bool.")

with open(own_path / 'Macros.yaml', 'r') as file:
    macros = yaml.safe_load(file)

# Selfmade modules. Some of these may need the config dictionary so it has to updated before importing/calling them
from CustomWidgets import (
    InfoSettingsButton,
    DragDropTableWidget,
    ConsoleWidget,
    ClickableMenu,
    EvaluixConsole,
    ExportTableWidget,
    InfoSettingsDialog,
    ProfileMacrosDialog,
    JumpSlider,
    MyMplCanvas,
    MacroStackedWidgetContainer,
    ManualDataDialog,
    HDF5PreviewDialog,
    ResultsTable,
    FitReportTable,
    FitPointsTable,
    FunctionViewer,
)
from FileLoader import read_file, Dataset, Data
data = Data()
from EvaluationFunctions import *

#TODO: Create just one update function for everything (table_metadata, table_datapkg, table_data, tabl
# e_loglist, canvas) and call it in the respective functions
# If programs slows down, check other solutions
#TODO: Write function so that the interaction with the gui is given by matching Widget names so that the only input is the main Tab/window name
# For example: locate_selected_data(tab) which returns the internal_id, data_pkg, data_cols, chosen_xdata or a part of this
# Maybe this function could also check if the selection changes and then update the respective widgets so that the update function is only called if necessary or desired

#Both of these TODO's could clean up the code a lot, make it more readable and easier to maintain and mainly extandable to other tabs which are designed in the same way

class MainWindow(QMainWindow):

    ###########################################
    # setup the main window
    ###########################################

    def __init__(self):
        super().__init__()

        # Load the .ui file
        uic.loadUi(ui_path, self)

        #######################################
        # General
        #######################################

        ############ Menu Bar ############
        
        # Load data
        self.load_afm_data = self.findChild(QAction, 'load_afm_data')
        self.load_afm_data.triggered.connect(
            lambda: self.safe_execute(
                lambda: (
                    read_file(data, file_or_dir=self.open_file_dialog(), dataformat='AFM'),
                    self.update_table_metadata(self.table_view_afm, type='AFM'),
                    self.save_progress() if ProfileConfig['Autosave_status'] else None,
                )
            )
        )
        
        self.load_lmoke_data = self.findChild(QAction, 'load_lmoke_data')
        self.load_lmoke_data.triggered.connect(
            lambda: self.safe_execute(
                lambda: (
                    read_file(data, file_or_dir=self.open_file_dialog(), dataformat='MOKE'),
                    self.update_table_metadata(self.hys_table_metadata, type='Hyst'),
                    self.save_progress() if ProfileConfig['Autosave_status'] else None,
                )
            )
        )
        
        self.load_vmoke_data = self.findChild(QAction, 'load_vmoke_data')
        self.load_vmoke_data.triggered.connect(
            lambda: self.safe_execute(
                lambda: (
                    read_file(data, file_or_dir=self.open_file_dialog(), dataformat='MOKE'),
                    self.update_table_metadata(self.hys_table_metadata, type='Hyst'),
                    self.save_progress() if ProfileConfig['Autosave_status'] else None,
                )
            )
        )

        self.load_vsm_data = self.findChild(QAction, 'load_vsm_data')
        self.load_vsm_data.triggered.connect(
            lambda: self.safe_execute(
                lambda: (
                    read_file(data, file_or_dir=self.open_file_dialog(), dataformat='VSM'),
                    self.update_table_metadata(self.hys_table_metadata, type='Hyst'),
                    self.save_progress() if ProfileConfig['Autosave_status'] else None,
                )
            )
        )
        
        self.load_kerr_imgs = self.findChild(QAction, 'load_kerr_imgs')
        self.load_kerr_imgs.triggered.connect(
            lambda: self.safe_execute(
                lambda: (
                    read_file(data, file_or_dir=self.open_file_dialog(), dataformat='Kerr_imgs'),
                    self.update_table_metadata(self.hys_table_metadata, type='Hyst'),
                    self.save_progress() if ProfileConfig['Autosave_status'] else None,
                )
            )
        )
        
        self.load_kerr_hys = self.findChild(QAction, 'load_kerr_hys')
        self.load_kerr_hys.triggered.connect(
            lambda: self.safe_execute(
                lambda: (
                    read_file(data, file_or_dir=self.open_file_dialog(), dataformat='Kerr'),
                    self.update_table_metadata(self.hys_table_metadata, type='Hyst'),
                    self.save_progress() if ProfileConfig['Autosave_status'] else None,
                )
            )
        )
        
        # Save evaluation progress
        self.save_path = None

        self.Save_progress_as_hdf5 = self.findChild(QAction, 'Save_progress_as_hdf5')
        self.Save_progress_as_hdf5.triggered.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.save_progress(
                        save_path=self.open_file_dialog(intention='save', filter='HDF5 files (*.h5)'),
                        ids=None),
                )
            )
        )

        self.Toggle_Autosave_Progress = self.findChild(QAction, 'Toggle_Autosave_Progress')
        self.Toggle_Autosave_Progress.triggered.connect(self.toggle_autosave_status)


        # Load evaluation progress
        self.load_progress_with_preview = self.findChild(QAction, 'load_progress_with_preview')
        self.load_progress_with_preview.triggered.connect(
            lambda: self.safe_execute(
                lambda: (self.load_progress(preview=True),
                        self.hys_btn_update_all.click(),
                )
            )
        )
        
        self.load_progress_directly = self.findChild(QAction, 'load_progress_directly')
        self.load_progress_directly.triggered.connect(
            lambda: self.safe_execute(
                lambda: (self.load_progress(preview=False),
                        self.hys_btn_update_all.click(),
                )
            )
        )
        
        self.menuFunctions = self.findChild(ClickableMenu, 'menuFunctions')
        self.menuFunctions.clicked.connect(
            self.open_FunctionViewer
        )
        
        # Insane Clown Posse
        self.actionActualHelp = self.findChild(QAction, 'actionActualHelp')
        
        # Connect the QAction to a slot that opens the YouTube link
        self.actionActualHelp.triggered.connect(self.ICP)

        ############ Status Bar ############

        self.statusbar = self.findChild(QStatusBar, 'statusbar')
        self.setStatusBar(self.statusbar)

        # Create labels for the status bar
        self.status_dataset_label = QLabel()
        self.status_autosave_label = QLabel()
        self.status_save_label = QLabel()
        self.status_message_label = QLabel()

        # Add the labels to the status bar
        self.statusbar.addWidget(self.status_dataset_label)
        self.statusbar.addWidget(self.status_autosave_label)
        self.statusbar.addWidget(self.status_save_label)
        self.statusbar.addWidget(self.status_message_label)

        #######################################
        # General about the TabWidget
        #######################################
        self.tab_main = self.findChild(QTabWidget, 'tab_main')
        self.tab_main.currentChanged.connect(self.update_all_tab)

        #######################################
        # AFM Tab
        # get all widgets of the AFM tab
        #######################################

        # # QTableView which vizualizes the metadata of data with type Img (currently just AFM)
        # self.table_view_afm = self.findChild(QTabWidget, 'table_view_afm')
        # # initialize the table_view_afm
        # self.update_table_metadata(self.table_view_afm, type='Img')

        # # QPushButton which triggers the update of the table_view_afm
        # self.btn_update_table_view_afm = self.findChild(QPushButton, 'btn_update_table_view_afm')
        # self.btn_update_table_view_afm.clicked.connect(
        #     lambda: self.safe_execute(
            # lambda: self.update_table_metadata(self.table_view_afm, type='AFM')
        #     )
        #   )

        #######################################
        # Hysteresis Tab
        #######################################
        self.initialize_hys_tab()

        #######################################
        # Kerr Tab
        #######################################
        self.initialize_kerr_tab()

        #######################################
        # AFM Tab
        #######################################
        self.initialize_afm_tab()
        # self.afm_btn_update_all.click()

        #######################################
        # Console Tab
        #######################################
        self.initialize_console_tab()

        #######################################
        # Datamanager Tab
        #######################################
        self.initialize_datamanager_tab()

        # software starts with the Hysteresis tab
        self.tab_main.setCurrentIndex(0)
        self.hys_btn_update_all.click()

    def initialize_console_tab(self):
        self.console_widget = self.findChild(EvaluixConsole, 'console_widget')
        # self.console_widget.console.execute("\n".join([line.lstrip() for line in """
        #     import numpy as np
        #     import scipy
        #     import pandas as pd
        #     import matplotlib.pyplot as plt
        #     from FileLoader import data, read_file, Dataset
        #     from EvaluationFunctions import *
        #     """.split('\n')]))

        #TODO: Check if this is going to be supported in PyQt6
        # # Find the QTextEdit widget and its layout
        # self.console_text_placeholder = self.findChild(QTextEdit, 'console_text_placeholder')
        # layout = self.console_text_placeholder.parent().layout()

        # # Create a new QsciScintilla widget
        # self.editor = QsciScintilla()

        # # Add a few features to the editor
        # self.lexer = QsciLexerPython()
        # self.editor.setLexer(lexer)
        # self.editor.setAutoIndent(True)

        # # Replace the QTextEdit widget with the QsciScintilla widget
        # layout.replaceWidget(self.console_text_placeholder, self.editor)

    def initialize_datamanager_tab(self):
        #######################################
        # Datamanager Tab
        #######################################

        ############ Metadata table ############
        self.dm_table_metadata = self.findChild(DragDropTableWidget, 'dm_table_metadata')
        
        self.dm_table_metadata.selectionModel().selectionChanged.connect(
            lambda: self.safe_execute(
                lambda: self.dm_btn_update_all.click()
                )
            )
        
        self.dm_table_metadata.fileDropped.connect(
            lambda file_path: self.safe_execute(
                lambda: (
                    read_file(data, file_or_dir=file_path, dataformat=self.dm_load_dataformat.currentText()),
                    self.dm_btn_update_all.click(),
                    self.save_progress() if ProfileConfig['Autosave_status'] else None,
                    )
                )
            )
        # make the metadata editable
        self.dm_table_metadata.setEditTriggers(QTableView.EditTrigger.DoubleClicked)
        self.dm_table_metadata.itemChanged.connect(
            lambda item: self.safe_execute(
                lambda: (
                    self.change_metadata(item),
                    self.dm_btn_update_all.click()
                )
            )
        )
        
        # QPushButton which triggers the update of all dm tables
        self.dm_btn_update_all = self.findChild(QPushButton, 'dm_btn_update_all')
        self.dm_btn_update_all.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.update_all(type='All'),
                    )
                )
            )

        # QComboBox to select the type of data to be loaded
        self.dm_load_dataformat = self.findChild(QComboBox, 'dm_load_dataformat')

        self.dm_btn_load_data = self.findChild(QPushButton, 'dm_btn_load_data')
        self.dm_btn_load_data.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    read_file(data, file_or_dir=self.open_file_dialog(), dataformat=self.dm_load_dataformat.currentText()),
                    self.dm_btn_update_all.click(),
                    self.save_progress() if ProfileConfig['Autosave_status'] else None,
                    )
                )
            )

        self.dm_del_dataset = self.findChild(QPushButton, 'dm_del_dataset')
        self.dm_del_dataset.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.del_dataset(),
                    self.dm_btn_update_all.click(),
                    )
                )
            )
        
        self.dm_add_data_manually = self.findChild(QPushButton, 'dm_add_data_manually')
        self.dm_add_data_manually.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    print('add data manually'),
                    self.add_data_manually(),
                    self.dm_btn_update_all.click(),
                    )
                )
            )

    def initialize_hys_tab(self):
        #######################################
        # Hysteresis Tab
        #######################################

        ############ Metadata table ############
        # QTableView which vizualizes the metadata for all loaded data with type Hyst
        self.hys_table_metadata = self.findChild(DragDropTableWidget, 'hys_table_metadata')

        # connect the current selection of the hys_table_metadata to the function visualize_selected_data
        self.hys_table_metadata.selectionModel().selectionChanged.connect(
            lambda: self.safe_execute(
                lambda: self.hys_btn_update_all.click()
                )
            )
        self.hys_table_metadata.fileDropped.connect(
            lambda file_path: self.safe_execute(
                lambda: (
                    read_file(data, file_or_dir=file_path, dataformat=self.hys_load_dataformat.currentText()),
                    self.hys_btn_update_all.click(),
                    self.save_progress() if ProfileConfig['Autosave_status'] else None,
                )
            )
        )

        # QPushButton which triggers the update of all hys tables
        self.hys_btn_update_all = self.findChild(QPushButton, 'hys_btn_update_all')
        self.hys_btn_update_all.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.update_all(type='Hyst')
                    )
                )
            )

        # QComboBox to select the type of data to be loaded
        self.hys_load_dataformat = self.findChild(QComboBox, 'hys_load_dataformat')

        # QPushButton which loads new data
        self.hys_btn_load_data = self.findChild(QPushButton, 'hys_btn_load_data')
        self.hys_btn_load_data.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    read_file(data, file_or_dir=self.open_file_dialog(), dataformat=self.hys_load_dataformat.currentText()),
                    self.update_table_metadata(self.hys_table_metadata, type='Hyst'),
                    self.save_progress() if ProfileConfig['Autosave_status'] else None,
                )
            )
        )

        # QPushButton which triggers the deletion of the selected dataset
        self.hys_del_dataset = self.findChild(QPushButton, 'hys_del_dataset')
        self.hys_del_dataset.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.del_dataset(),
                    self.hys_btn_update_all.click(),
                    )
                )
            )

        ############ Data Panel ############

        # QGroupBox parent which is the container for the canvas and hys_table_datapkg and hys_table_data
        self.hys_data_panel = self.findChild(QGroupBox, 'hys_data_panel')

        # Canvas, toolbar and table
        # QWidget which is the placeholder for the canvas
        self.hys_canvas_placeholder = self.findChild(QWidget, 'hys_canvas_placeholder')
        self.hys_data_panel.canvas = MyMplCanvas(self.hys_canvas_placeholder, Purpose="Hysteresis")
        self.hys_data_panel.canvas.toolbar.setStyleSheet("""
            QToolBar { background-color : rgb(45, 55, 65); color : white; }
            QToolButton { background-color : white; color : black; }
        """)

        # Override the default layout of the placeholder, insert the canvas and the toolbar
        layout = QVBoxLayout()
        layout.addWidget(self.hys_data_panel.canvas.toolbar)
        layout.addWidget(self.hys_data_panel.canvas)
        self.hys_canvas_placeholder.setLayout(layout)
        self.hys_data_panel.canvas.figure.tight_layout()

        # Add the QTableWidget to the hys_data_panel
        self.hys_data_panel.table = self.findChild(QTableWidget, 'hys_table_plot_content')

        # QPushButton to delete the selected data from preserved data packages
        self.hys_del_preserved_content = self.findChild(QPushButton, 'hys_del_preserved_content')
        self.hys_del_preserved_content.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.del_preserved_content(self.hys_data_panel),
                    self.hys_btn_update_all.click(),
                    )
                )
            )

        # QComboBox to select the x-data for the data table
        self.hys_data_xdata = self.findChild(QComboBox, 'hys_data_xdata')
        #self.update_choose_xdata(self.hys_data_xdata)
        self.hys_data_xdata.currentIndexChanged.connect(
            lambda: self.safe_execute(
                lambda: self.visualize_selected_data(self.hys_data_panel)
                )
            )

        # QPushButton to preserve the currently plotted data packages
        self.hys_preserve_current_datapgk = self.findChild(QPushButton, 'hys_preserve_current_datapgk')
        self.hys_preserve_current_datapgk.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.preserve_current_datapgk(self.hys_data_panel),
                    self.hys_btn_update_all.click(),
                    )
                )
            )

        # hys_table_data
        # QTableView to vizualize the data of a selected data in a data package
        self.hys_table_data = self.findChild(QTableWidget, 'hys_table_data')
        self.hys_table_data.prev_selection = []
        #self.update_table_data(self.hys_table_data)
        self.hys_table_data.selectionModel().selectionChanged.connect(
            lambda: (
                self.safe_execute(
                    lambda: self.check_fit_selection(self.hys_table_data)
                ),
                self.safe_execute(
                    lambda: self.visualize_selected_data(self.hys_data_panel)
                ),
            )
        )

        ############ Control Panel ############

        # QPush which triggers the deletion of outliers in the selected data
        self.btn_hys_del_outliers = self.findChild(InfoSettingsButton, 'btn_hys_del_outliers')
        self.btn_hys_del_outliers.buttonClicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.data_manipulation(del_outliers, add_xdata=False),
                    self.hys_btn_update_all.click())
                )
            )
        self.btn_hys_del_outliers.infoClicked.connect(
            lambda: self.safe_execute(
                lambda: self.info_settings(del_outliers)
                )
            )

        # QPushButton which triggers the opening removal of the selected data
        self.btn_hys_rmv_opening = self.findChild(InfoSettingsButton, 'btn_hys_rmv_opening')
        self.btn_hys_rmv_opening.buttonClicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.data_manipulation(rmv_opening, add_xdata=False),
                    self.hys_btn_update_all.click())
                )
            )
        self.btn_hys_rmv_opening.infoClicked.connect(
            lambda: self.safe_execute(
                lambda: self.info_settings(rmv_opening)
                )
            )

        # QPushButton which triggers the slope correction of the selected data
        self.btn_hys_slope = self.findChild(InfoSettingsButton, 'btn_hys_slope')
        self.btn_hys_slope.buttonClicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.data_manipulation(slope_correction, add_xdata=True),
                    self.hys_btn_update_all.click())
                )
            )
        self.btn_hys_slope.infoClicked.connect(
            lambda: self.safe_execute(
                lambda: self.info_settings(slope_correction)
                )
            )

        # QPushButton which triggers the normalization of the selected data
        self.btn_hys_norm = self.findChild(InfoSettingsButton, 'btn_hys_norm')
        self.btn_hys_norm.buttonClicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.data_manipulation(hys_norm, add_xdata=True),
                    self.hys_btn_update_all.click())
            )
            )
        self.btn_hys_norm.infoClicked.connect(
            lambda: self.safe_execute(
                lambda: self.info_settings(hys_norm)
                )
            )
        
        # QPushButton which triggers the centering of the selected data
        self.btn_hys_center = self.findChild(InfoSettingsButton, 'btn_hys_center_signal')
        self.btn_hys_center.buttonClicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.data_manipulation(hys_center, add_xdata=True),
                    self.hys_btn_update_all.click())
            )
            )
        self.btn_hys_center.infoClicked.connect(
            lambda: self.safe_execute(
                lambda: self.info_settings(hys_center)
                )
            )

        # QPushButton which triggers the normalization of the selected data
        self.btn_hys_smooth = self.findChild(InfoSettingsButton, 'btn_hys_smooth')
        self.btn_hys_smooth.buttonClicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.data_manipulation(smoothing1d, add_xdata=False),
                    self.hys_btn_update_all.click())
                )
            )
        self.btn_hys_smooth.infoClicked.connect(
            lambda: self.safe_execute(
                lambda: self.info_settings(smoothing1d)
                )
            )

        # QPushButton which triggers the numerical differentiation of the selected data
        self.btn_hys_num_deriv = self.findChild(InfoSettingsButton, 'btn_hys_num_deriv')
        self.btn_hys_num_deriv.buttonClicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.data_manipulation(num_derivative, add_xdata=True),
                    self.hys_btn_update_all.click())
                )
            )
        self.btn_hys_num_deriv.infoClicked.connect(
            lambda: self.safe_execute(
                lambda: self.info_settings(num_derivative)
                )
            )

        # QPushButton which triggers the numerical integration of the selected data
        self.btn_hys_integrate = self.findChild(InfoSettingsButton, 'btn_hys_integrate')
        self.btn_hys_integrate.buttonClicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.data_manipulation(num_integral, add_xdata=True),
                    self.hys_btn_update_all.click())
                )
            )
        self.btn_hys_integrate.infoClicked.connect(
            lambda: self.safe_execute(
                lambda: self.info_settings(num_integral)
                )
            )

        # QPushButton which triggers the numerical integration of the selected data
        self.btn_hys_num_integ = self.findChild(InfoSettingsButton, 'btn_hys_num_integ')


        # QPushButton which triggers the linear fit of the selected data
        self.btn_hys_lin_fit = self.findChild(InfoSettingsButton, 'btn_hys_lin_fit')
        self.btn_hys_lin_fit.buttonClicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.data_evaluation(lin_hyseval, add_xdata=True),
                    self.check_fit_selection(self.hys_table_data, fit=True),
                    self.hys_btn_update_all.click())
                )
            )
        self.btn_hys_lin_fit.infoClicked.connect(
            lambda: self.safe_execute(
                lambda: self.info_settings(lin_hyseval)
                )
            )

        # QPushButton which triggers the arctan fit of the selected data
        self.btn_hys_arctan_fit = self.findChild(InfoSettingsButton, 'btn_hys_arctan_fit')
        self.btn_hys_arctan_fit.buttonClicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.data_evaluation(tan_hyseval, add_xdata=True),
                    self.check_fit_selection(self.hys_table_data, fit=True),
                    self.hys_btn_update_all.click())
                )
            )
        self.btn_hys_arctan_fit.infoClicked.connect(
            lambda: self.safe_execute(
                lambda: self.info_settings(tan_hyseval)
                )
            )

        # QPushButton which triggers the double arctan fit of the selected data
        self.btn_hys_darctan_fit = self.findChild(InfoSettingsButton, 'btn_hys_darctan_fit')
        self.btn_hys_darctan_fit.buttonClicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.data_evaluation(double_tan_hyseval, add_xdata=True),
                    self.check_fit_selection(self.hys_table_data, fit=True),
                    self.hys_btn_update_all.click())
                )
            )
        self.btn_hys_darctan_fit.infoClicked.connect(
            lambda: self.safe_execute(
                lambda: self.info_settings(double_tan_hyseval)
                )
            )
        
        self.hys_macro_widget = self.findChild(MacroStackedWidgetContainer, 'hys_macro_widget')
        #self.hys_macro_widget.editButton.clicked.connect(ProfileMacrosDialog(self, macros).exec_)
        #self.hys_macro_widget.predefined_macros()
        
        # QListWidget which contains the loglist of the selected data package
        self.hys_loglist = self.findChild(QListWidget, 'hys_loglist')
        #self.update_loglist(self.hys_loglist)


        ############ Datapkg sub-Panel ############

        # hys_table_datapkg
        # QTableWidget to vizualize an overview about the data of the currently selected data package
        self.hys_table_datapkg = self.findChild(QTableWidget, 'hys_table_datapkg')
        #self.update_table_datapkg(self.hys_table_datapkg)
        self.hys_table_datapkg.selectionModel().selectionChanged.connect(
            lambda: self.safe_execute(
                lambda: self.hys_btn_update_all.click()
                )
            )


        # QCheckBox to trigger the creation of a new data package
        self.hys_new_data_flag = self.findChild(QCheckBox, 'hys_new_data_flag')

        # QPushButton which triggers the deletion of the selected data package
        self.hys_del_mod_data = self.findChild(QPushButton, 'hys_del_mod_data')
        self.hys_del_mod_data.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.del_mod_data(),
                    self.hys_btn_update_all.click(),
                    )
                )
            )

        # button to copy the selected data package
        self.hys_copy_datapkg = self.findChild(QPushButton, 'hys_copy_datapkg')
        self.hys_copy_datapkg.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.copy_mod_data(),
                    self.hys_btn_update_all.click(),
                    )
                )
            )

        # button to copy the selected data package column
        self.hys_copy_datacol = self.findChild(QPushButton, 'hys_copy_datacol')
        self.hys_copy_datacol.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.copy_datacol(),
                    self.hys_btn_update_all.click(),
                    )
                )
            )


        ############ Results sub-Panel ############
        self.hys_results_tab = self.findChild(QTabWidget, 'hys_results_tab')

        self.hys_choose_evalmodel = self.findChild(QComboBox, 'hys_choose_evalmodel')
        self.hys_choose_evalmodel.currentIndexChanged.connect(
            lambda: self.safe_execute(
                lambda: self.update_results_tab(self.hys_results_tab)
                )
            )
        

        self.update_results_tab(self.hys_results_tab)

        # # At the end of the initialization, update/initialize all widgets
        # self.hys_btn_update_all.click()

    def initialize_kerr_tab(self):

        ############ Metadata table ############
        # QTableView which vizualizes the metadata for all loaded data with type Hyst
        self.kerr_table_metadata = self.findChild(DragDropTableWidget, 'kerr_table_metadata')

        # connect the current selection of the hys_table_metadata to the function visualize_selected_data
        self.kerr_table_metadata.selectionModel().selectionChanged.connect(
            lambda: self.safe_execute(
                lambda: self.kerr_btn_update_all.click()
                )
            )
        self.kerr_table_metadata.fileDropped.connect(
            lambda file_path: self.safe_execute(
                lambda: (
                    read_file(data, file_or_dir=file_path, dataformat=self.kerr_load_dataformat.currentText()),
                    self.kerr_btn_update_all.click(),
                    self.save_progress() if ProfileConfig['Autosave_status'] else None,
                )
            )
        )

        # QPushButton which triggers the update of all hys tables
        self.kerr_btn_update_all = self.findChild(QPushButton, 'kerr_btn_update_all')
        self.kerr_btn_update_all.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.update_all(type='Img')
                    )
                )
            )

        # QComboBox to select the type of data to be loaded
        self.kerr_load_dataformat = self.findChild(QComboBox, 'kerr_load_dataformat')

        # QPushButton which loads new data
        self.kerr_btn_load_data = self.findChild(QPushButton, 'kerr_btn_load_data')
        self.kerr_btn_load_data.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    read_file(data, file_or_dir=self.open_file_dialog(), dataformat=self.kerr_load_dataformat.currentText()),
                    self.update_table_metadata(self.kerr_table_metadata, type='Img'),
                    self.save_progress() if ProfileConfig['Autosave_status'] else None,
                )
            )
        )

        # QPushButton which triggers the deletion of the selected dataset
        self.kerr_del_dataset = self.findChild(QPushButton, 'kerr_del_dataset')
        self.kerr_del_dataset.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.del_dataset(),
                    self.kerr_btn_update_all.click(),
                    )
                )
            )

        ############ Data Panel ############

        # QGroupBox parent which is the container for the canvas and hys_table_datapkg and hys_table_data
        self.kerr_data_panel = self.findChild(QGroupBox, 'kerr_data_panel')

        # Canvas, toolbar and table
        # QWidget which is the placeholder for the canvas
        self.kerr_canvas_placeholder = self.findChild(QWidget, 'kerr_canvas_placeholder')
        self.kerr_data_panel.canvas = MyMplCanvas(self.hys_canvas_placeholder, Purpose="Image")
        self.kerr_data_panel.canvas.toolbar.setStyleSheet("""
            QToolBar { background-color : rgb(45, 55, 65); color : white; }
            QToolButton { background-color : white; color : black; }
        """)

        # Override the default layout of the placeholder, insert the canvas and the toolbar
        layout = QVBoxLayout()
        layout.addWidget(self.kerr_data_panel.canvas.toolbar)
        layout.addWidget(self.kerr_data_panel.canvas)
        self.kerr_canvas_placeholder.setLayout(layout)
        self.kerr_data_panel.canvas.figure.tight_layout()

        # QSlider and QLabel to select the image to be displayed
        self.kerr_data_slider = self.findChild(JumpSlider, 'kerr_data_slider')
        self.kerr_data_slider.valueChanged.connect(
            lambda: self.safe_execute(
                lambda: (self.update_sliderlabels(),
                self.visualize_selected_imgs(self.kerr_data_panel))
                )
            )
        
        # # Add the QTableWidget to the hys_data_panel
        # self.hys_data_panel.table = self.findChild(QTableWidget, 'hys_table_plot_content')

        # # QPushButton to delete the selected data from preserved data packages
        # self.hys_del_preserved_content = self.findChild(QPushButton, 'hys_del_preserved_content')
        # self.hys_del_preserved_content.clicked.connect(
        #     lambda: self.safe_execute(
        #         lambda: (
        #             self.del_preserved_content(self.hys_data_panel),
        #             self.hys_btn_update_all.click(),
        #             )
        #         )
        #     )

        # # QComboBox to select the x-data for the data table
        # self.hys_data_xdata = self.findChild(QComboBox, 'hys_data_xdata')
        # #self.update_choose_xdata(self.hys_data_xdata)
        # self.hys_data_xdata.currentIndexChanged.connect(
        #     lambda: self.safe_execute(
        #         lambda: self.visualize_selected_data(self.hys_data_panel)
        #         )
        #     )

        # # QPushButton to preserve the currently plotted data packages
        # self.hys_preserve_current_datapgk = self.findChild(QPushButton, 'hys_preserve_current_datapgk')
        # self.hys_preserve_current_datapgk.clicked.connect(
        #     lambda: self.safe_execute(
        #         lambda: (
        #             self.preserve_current_datapgk(self.hys_data_panel),
        #             self.hys_btn_update_all.click(),
        #             )
        #         )
        #     )

        ############ Datapkg sub-Panel ############

        # QTableWidget to vizualize an overview about the data of the currently selected data package
        self.kerr_table_datapkg = self.findChild(QTableWidget, 'kerr_table_datapkg')
        self.kerr_table_datapkg.selectionModel().selectionChanged.connect(
            lambda: self.safe_execute(
                lambda: self.kerr_btn_update_all.click()
                )
            )

        # hys_table_data
        # QTableView to vizualize the data of a selected data in a data package
        self.kerr_table_data = self.findChild(QTableWidget, 'kerr_table_data')
        self.kerr_table_data.selectionModel().selectionChanged.connect(
            lambda: self.safe_execute(
                lambda: self.visualize_selected_imgs(self.kerr_data_panel)
                )
            )
        
        # connect the data_slider (initialized above) and the data_table
        self.kerr_selection_changed = False # flag to prevent the slider_to_table function to be called when the table is updated, i.e. selection cascades
        # Connect the signals and slots
        self.kerr_data_slider.valueChanged.connect(lambda value: self.slider_to_table(self.kerr_table_data, value))
        self.kerr_table_data.itemSelectionChanged.connect(lambda: self.table_to_slider(self.kerr_data_slider, self.kerr_table_data.currentRow()))


        # QCheckBox to trigger the creation of a new data package
        self.kerr_new_data_flag = self.findChild(QCheckBox, 'kerr_new_data_flag')

        # QPushButton which triggers the deletion of the selected data package
        self.kerr_del_mod_data = self.findChild(QPushButton, 'kerr_del_mod_data')
        self.kerr_del_mod_data.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.del_mod_data(),
                    self.kerr_btn_update_all.click(),
                    )
                )
            )

        # button to copy the selected data package
        self.kerr_copy_datapkg = self.findChild(QPushButton, 'kerr_copy_datapkg')
        self.kerr_copy_datapkg.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.copy_mod_data(),
                    self.kerr_btn_update_all.click(),
                    )
                )
            )

        # button to copy the selected data package column
        self.kerr_copy_datacol = self.findChild(QPushButton, 'kerr_copy_datacol')
        self.kerr_copy_datacol.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.copy_datacol(),
                    self.kerr_btn_update_all.click(),
                    )
                )
            )

    def initialize_afm_tab(self):
        #######################################
        # AFM Tab
        #######################################

        ############ Metadata table ############
        # QTableView which vizualizes the metadata for all loaded data with type AFM
        self.afm_table_metadata = self.findChild(DragDropTableWidget, 'afm_table_metadata')

        # connect the current selection of the hys_table_metadata to the function visualize_selected_data
        self.afm_table_metadata.selectionModel().selectionChanged.connect(
            lambda: self.safe_execute(
                lambda: self.afm_btn_update_all.click()
                )
            )
        self.afm_table_metadata.fileDropped.connect(
            lambda file_path: self.safe_execute(
                lambda: (
                    read_file(data, file_or_dir=file_path, dataformat=self.afm_load_dataformat.currentText()),
                    self.afm_btn_update_all.click(),
                    self.save_progress() if ProfileConfig['Autosave_status'] else None,
                )
            )
        )

        # QPushButton which triggers the update of all hys tables
        self.afm_btn_update_all = self.findChild(QPushButton, 'afm_btn_update_all')
        self.afm_btn_update_all.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.update_all(type='AFM')
                    )
                )
            )

        # QComboBox to select the type of data to be loaded
        self.afm_load_dataformat = self.findChild(QComboBox, 'afm_load_dataformat')

        # QPushButton which loads new data
        self.afm_btn_load_data = self.findChild(QPushButton, 'afm_btn_load_data')
        self.afm_btn_load_data.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    read_file(data, file_or_dir=self.open_file_dialog(), dataformat=self.afm_load_dataformat.currentText()),
                    self.afm_btn_update_all.click(),
                    self.save_progress() if ProfileConfig['Autosave_status'] else None,
                )
            )
        )

        # QPushButton which triggers the deletion of the selected dataset
        self.afm_del_dataset = self.findChild(QPushButton, 'afm_del_dataset')
        self.afm_del_dataset.clicked.connect(
            lambda: self.safe_execute(
                lambda: (
                    self.del_dataset(),
                    self.afm_btn_update_all.click(),
                    )
                )
            )

        ############ Data Panel ############

        # QGroupBox parent which is the container for the canvas
        self.afm_data_panel = self.findChild(QGroupBox, 'afm_data_panel')

        # Canvas, toolbar and table
        # QWidget which is the placeholder for the canvas
        self.afm_canvas_placeholder = self.findChild(QWidget, 'afm_canvas_placeholder')
        self.afm_data_panel.canvas = MyMplCanvas(self.afm_canvas_placeholder, Purpose="AFM")
        self.afm_data_panel.canvas.toolbar.setStyleSheet("""
            QToolBar { background-color : rgb(45, 55, 65); color : white; }
            QToolButton { background-color : white; color : black; }
        """)

        # Override the default layout of the placeholder, insert the canvas and the toolbar
        layout = QVBoxLayout()
        layout.addWidget(self.afm_data_panel.canvas.toolbar)
        layout.addWidget(self.afm_data_panel.canvas)
        self.afm_canvas_placeholder.setLayout(layout)
        self.afm_data_panel.canvas.figure.tight_layout()


        # at the end of the initialization, update/initialize all widgets
        self.afm_btn_update_all.click()

    ###########################################
    # main functions for the main window
    ###########################################
    # 0. Safe execution of functions
    # 1. Load data window
    # 2. Update and connect the widgets and status bar (order: metadata, datapkg, data, xdata, loglist, statusbar)
    # 3. get and delete selected data
    # 4. Data visualization, manipulation and evaluation

    # 0. Safe execution of functions
    def safe_execute(self, func):
        try:
            #result = func()
            func()
        except Exception as e:
            error_message = str(e) + "\n\n" + traceback.format_exc()

            msgBox = QMessageBox(self)
            msgBox.setIcon(QMessageBox.Icon.Critical)
            msgBox.setText("Error")
            msgBox.setInformativeText(str(e))

            # Set the detailed text
            msgBox.setDetailedText(error_message)

            # Add a "Copy to Clipboard" button
            copyButton = msgBox.addButton("Copy details to Clipboard", QMessageBox.ButtonRole.ActionRole)

            # Disconnect the button's signals
            try:
                copyButton.clicked.disconnect()
            except TypeError:
                pass  # No signals were connected
            copyButton.clicked.connect(lambda: self.copy_to_clipboard(error_message))

            # Add a "OK" button
            okButton = msgBox.addButton(QMessageBox.StandardButton.Ok)

            msgBox.exec()

        # return result

    def copy_to_clipboard(self, text):
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

    # 1. Load data window
    def open_file_dialog(self, intention='load', filter='All Files (*)'):
        if intention == 'load':
            # Open a file dialog to choose a file/directory
            file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", filter)
            if file_name:
                return file_name
            return None
        elif intention == 'save':
            # Open a file dialog to choose a file/directoryconda install -c conda-forge opencv
            file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "", filter)
            if file_name:
                return file_name
            return None
        else:
            self.update_status_bar("Invalid intention for loading or saving files. Please choose 'load' or 'save' for open_file_dialog().")

    # 2. Update and connect the widgets and status bar
    def update_table_metadata(self, table_metadata, type):
        # Check if the data object is empty
        if not data.dataset:
            self.update_status_bar("No data available. Please load a dataset.")
            return
        else:
            current_status = self.statusBar().currentMessage()
            if current_status == "No data available. Please load a dataset.":
                self.update_status_bar(current_status)

        key_importance = {
            'internal_id': 0,
            'sample': 1,
            'type': 2,
            'device': 3,
            'datetime': 4,
            'comment': 5,
            'uuid': 6,
        }

        # List to store the horizontal and vertical headers
        horizontal_headers = []
        vertical_headers = []

        # Prepare a list to store the table data
        table_content = []

        if type == 'All':
            # Iterate over the items in the data dictionary
            for id, dataset in data.dataset.items():
                # Reading out the metadata
                metadata = dataset.metadata

                # Exclude 'internal_id' from the keys and add to horizontal_headers
                for key in metadata.keys():
                    if key not in horizontal_headers:
                        horizontal_headers.append(str(key))

                # Sort the keys based on their importance, if the key is not in the dictionary, it will be sorted to the end
                horizontal_headers.sort(key=lambda k: key_importance.get(k, float('inf')))

                # Prepare a row data
                row_data = [str(metadata.get(key, '-')) for key in horizontal_headers]
                table_content.append(row_data)

        else:
            # Iterate over the items in the data dictionary
            for id, dataset in data.dataset.items():
                # Check if the metadata contains the type
                if type in dataset.metadata.values():
                    # Reading out the metadata
                    metadata = dataset.metadata

                    # Exclude 'internal_id' from the keys and add to horizontal_headers
                    for key in metadata.keys():
                        if key not in horizontal_headers:
                            horizontal_headers.append(str(key))

                    # Sort the keys based on their importance, if the key is not in the dictionary, it will be sorted to the end
                    horizontal_headers.sort(key=lambda k: key_importance.get(k, float('inf')))

                    # Prepare a row data
                    row_data = [str(metadata.get(key, '-')) for key in horizontal_headers]
                    table_content.append(row_data)

        # Set the number of rows and columns
        table_metadata.setRowCount(len(table_content))
        table_metadata.setColumnCount(len(horizontal_headers))
        vertical_headers = [str(i) for i in range(len(table_content))]
        
        # Add data to the table_metadata
        for i, row_data in enumerate(table_content):
            for j, cell_data in enumerate(row_data):
                # try:
                #     num = float(cell_data)
                #     str_num = "{:.5f}".format(num)
                #     item = NumericTableWidgetItem(str_num.rstrip('0').rstrip('.') if '.' in str_num else str_num)
                # except:
                item = QTableWidgetItem(cell_data)
                table_metadata.setItem(i, j, item)

        # substitute the 'internal_id' with "ID" in the horizontal_headers, just for better readability
        try:
            horizontal_headers[horizontal_headers.index('internal_id')] = 'ID'
        except ValueError:
            pass

        # Set horizontal and vertical headers
        table_metadata.setHorizontalHeaderLabels(horizontal_headers)
        table_metadata.setVerticalHeaderLabels(vertical_headers)

        # make sure the table is updated
        table_metadata.resizeColumnsToContents()
        table_metadata.update()

    def update_table_datapkg(self, table_datapkg):
        # Prepare a list to store the table data
        table_content = []

        # get the internal_id
        selected_widgets, selected_data = self.get_selection()
        _id, _pkg, _cols, _x = selected_data
        if _id is not None:

            # # check if an update is necessary
            # try:
            #     if table_datapkg.internal_id == _id:
            #         return
            # except:
            #     table_datapkg.internal_id = _id

            data_dict = vars(data.dataset[_id]) # Convert selected dataclass to a dictionary

            # Filter out 'metadata' and 'loglist'
            horizontal_headers = [str(key) for key in data_dict.keys() if not key in ['metadata', 'loglist'] and isinstance(data_dict[key], pd.DataFrame)]

            row_data = [f"Cols: {data_dict[key].shape[1]}, Rows: {data_dict[key].shape[0]}" for key in horizontal_headers]
            table_content.append(row_data)

            # Check if current selection would be outside of new table_content
            selection = table_datapkg.selectionModel().selectedIndexes()
            if selection:
                if selection[0].row() >= len(table_content):
                    table_datapkg.clearSelection()
                elif selection[0].column() >= len(horizontal_headers):
                    table_datapkg.clearSelection()

            # Set the number of rows and columns
            table_datapkg.setRowCount(len(table_content))
            table_datapkg.setColumnCount(len(horizontal_headers))

            for i, row_data in enumerate(table_content):
                for j, cell_data in enumerate(row_data):
                    item = QTableWidgetItem(cell_data)
                    table_datapkg.setItem(i, j, item)

            # Check if anything is already selected
            if not table_datapkg.selectionModel().selectedIndexes():
                # Select the "raw_data" column
                if "raw_data" in horizontal_headers:
                    table_datapkg.selectColumn(horizontal_headers.index("raw_data"))

        else:
            # Add a dummy row of data to make the table visible and selectable
            table_datapkg.setRowCount(1)
            table_datapkg.setColumnCount(1)
            horizontal_headers = ['No data selected']
            item = QTableWidgetItem('Cols: 0, Rows: 0')
            table_datapkg.setItem(0, 0, item)

        # Set horizontal headers and update the table
        table_datapkg.setHorizontalHeaderLabels(horizontal_headers)
        table_datapkg.resizeColumnsToContents()
        table_datapkg.update()

    def update_table_data(self, table_data):

        # get the internal_id and the data package
        selected_widgets, selected_data = self.get_selection()
        _id, _pkg, _, _ = selected_data
        _, _, _, _, data_xdata = selected_widgets
        if _id is not None and _pkg:
            # Control the header of the (first) selected column
            if _pkg == 'No data selected':
                return

            _data = getattr(data.dataset[_id], _pkg)
            horizontal_headers = list(_data.keys())

            if isinstance(_data, pd.DataFrame):
                # Set the number of rows and columns
                table_data.setRowCount(_data.shape[0] + 1)
                table_data.setColumnCount(_data.shape[1])

                # Add the attribute "unit" to the first row
                for j in range(_data.shape[1]):
                    header = horizontal_headers[j]
                    # Attempt to get the unit attribute
                    if hasattr(_data[header], 'unit'):
                        unit = getattr(_data[header], 'unit', None)
                        quantity_type = None
                        for key in EvaluixConfig['conversion_factors']:
                            if str(unit) in EvaluixConfig['conversion_factors'][key].keys():
                                quantity_type = key
                                break
                        
                    else:
                        unit = 'unknown unit'  # or unit = ''
                        quantity_type = None
                    
                    # Create a QComboBox for unit selection
                    unit_selector = QComboBox()
                    if quantity_type is not None:
                        available_units = EvaluixConfig['conversion_factors'][quantity_type]
                    else:
                        available_units = [unit]
                    unit_selector.addItems(available_units)
                    unit_selector.setCurrentText(unit)
                    unit_selector.currentTextChanged.connect(lambda new_unit, header=header: 
                        self.update_unit(new_unit, 
                                         _data,
                                         header, 
                                         table_data,
                                         data_xdata))

                    table_data.setCellWidget(0, j, unit_selector)

                for i in range(_data.shape[0]):
                    for j in range(_data.shape[1]):
                        # check if header is 'FlattendImage'
                        if horizontal_headers[j] == 'FlattendImage':
                            item = QTableWidgetItem("Image")
                        elif isinstance(_data.iloc[i, j], (int, float)):
                            item = QTableWidgetItem("{:.5f}".format(_data.iloc[i, j]))
                        else:
                            item = QTableWidgetItem(str(_data.iloc[i, j]))
                        table_data.setItem(i+1, j, item) # i is the row, j is the column

            # Set horizontal headers and update the table
            table_data.setHorizontalHeaderLabels(horizontal_headers)
            
            vertical_headers = ['Unit'] + [str(i) for i in range(_data.shape[0])]
            table_data.setVerticalHeaderLabels(vertical_headers)

            if not table_data.selectedIndexes() or len(table_data.selectedIndexes()) != table_data.rowCount() * len(set(index.column() for index in table_data.selectedIndexes())):
                # Try to select column 1, if it exists
                if table_data.columnCount() > 1:
                    table_data.selectColumn(1)
                # If column 1 doesn't exist, select column 0
                elif table_data.columnCount() > 0:
                    table_data.selectColumn(0)

        else:
            # Add a dummy row of data to make the table visible and selectable
            table_data.setRowCount(1)
            table_data.setColumnCount(1)
            horizontal_headers = ['No data selected']
            item = QTableWidgetItem('Cols: 0, Rows: 0')
            table_data.setItem(0, 0, item)
            # Set horizontal headers
            table_data.setHorizontalHeaderLabels(horizontal_headers)

        # Update the table
        table_data.resizeColumnsToContents()
        table_data.update()
        
    def update_unit(self, new_unit, data, header, table_data, data_xdata):
        # Copy units from the original series
        units = {key: (data[key].unit if key != header else new_unit) for key in data.keys()}
        
        # Get the currently selected columns
        data[header] = unit_converter(data[header], EvaluixConfig['conversion_factors'], new_unit)
        
        # Reassign the units to the series in data
        for key, unit in units.items():
            data[key].unit = unit
        
        # Refresh the table data
        self.update_table_data(table_data)
        self.update_choose_xdata(data_xdata)
        
        # Emit selectionChanged signal for table_data
        selection_model = table_data.selectionModel()
        if selection_model.hasSelection():
            selected_indexes = selection_model.selectedIndexes()
            selection = QItemSelection()
            for index in selected_indexes:
                selection.select(index, index)
            selection_model.selectionChanged.emit(selection, selection)
        

    def check_fit_selection(self, table_data, fit=False):
        # Temporarily disconnect the signals
        table_data.blockSignals(True)

        # delete the memory of the previous selection so that all columns are selected in case of a fit
        if fit:
            table_data.prev_selection = []
            print("Fit selected, resetting selection")

        # Get the currently selected columns
        selected_columns = table_data.selectionModel().selectedColumns()

        # List to hold all columns that need to be selected
        columns_to_select = []

        # For each selected column, check if there are other columns with headers that contain the selected header
        for column in selected_columns:
            if column not in table_data.prev_selection:
                header = table_data.horizontalHeaderItem(column.column()).text()

                # Get all column headers
                all_headers = [table_data.horizontalHeaderItem(i).text() for i in range(table_data.columnCount())]

                # Find headers that contain the selected header
                matching_headers = [h for h in all_headers if header.strip() in h.strip() and '_fit' in h.strip()]

                # Add the matching columns to the list
                for h in matching_headers:
                    index = all_headers.index(h)
                    columns_to_select.append(index)

        # Initialize a selection object
        selection = QtCore.QItemSelection()

        # Select all the columns in the list
        for column in columns_to_select:
            # Get the top and bottom index of the column, i.e. the whole column
            top = table_data.model().index(0, column)
            bottom = table_data.model().index(table_data.rowCount() - 1, column)
            # Add the selection to the selection object
            selection.select(top, bottom)

        # Select the columns in the list
        table_data.selectionModel().select(selection, QtCore.QItemSelectionModel.SelectionFlag.Select)

        # Update the table
        table_data.update()

        # Get the currently selected columns after the check
        selected_columns = table_data.selectionModel().selectedColumns()

        # Save the current selection
        table_data.prev_selection = selected_columns

        # Reconnect the signals
        table_data.blockSignals(False)

    def update_choose_xdata(self, combobox):
        # get the internal_id and the data package
        selected_widgets, selected_data = self.get_selection()
        _id, _pkg, _cols, _x = selected_data
        if _id is not None and _pkg:

            if _pkg == 'No data selected': # if no data is selected
                combobox.clear()
                combobox.addItem("No data selected")
                combobox.setCurrentIndex(0)
                return

            # retrieve the selected dataframe or more precisely the keys of the dataframe
            _data = getattr(data.dataset[_id], _pkg)
            horizontal_headers = list(_data.keys())

            if isinstance(_data, pd.DataFrame):
                # save old data and clear the combobox
                old_xdata = combobox.currentText()
                combobox.clear()

                combobox.addItem("Point number")
                items = ["Point number"]
                for header in horizontal_headers:
                    try:
                        item = header + " [" + _data[header].unit + "]"
                    except AttributeError:
                        item = header
                    items.append(item)
                    combobox.addItem(item)

                # set the old data as current text
                if old_xdata in horizontal_headers:
                    index = combobox.findText(old_xdata)
                else:
                    for item in items:
                        if item.startswith(("H", "field", "Field")):
                            index = combobox.findText(item)
                            break
                        else:
                            index = combobox.findText("Point number")

                combobox.setCurrentIndex(index)
        else:
            # Add a dummy row of data to make the table visible and selectable
            combobox.clear()
            combobox.addItem("No data selected")
            combobox.setCurrentIndex(0)

    def update_loglist(self, table_loglist):
        # get the internal_id and the data package
        selected_widgets, selected_data = self.get_selection()
        _id, _pkg, _cols, _x = selected_data
        if _id is not None and _pkg:

            if _pkg == 'No data selected' or _pkg == 'raw_data':
                table_loglist.clear()
                return

            key = _pkg.split("_")[-1]
            loglist = getattr(data.dataset[_id], "loglist_" + key)

            #input the loglist into the QListWidget
            table_loglist.clear()
            for log in loglist:
                table_loglist.addItem(log)

    def update_results_tab(self, results_tab=None):
        tab_main = self.findChild(QTabWidget, 'tab_main')

        # Get the currently open tab
        index = tab_main.currentIndex()
        tab_name = tab_main.widget(index).objectName()

        # Identify tabs function by slicing the tab_name
        _name = tab_name.split("_")[-1] + "_"  # the "_" is important due to current naming convention

        if results_tab is None:
            results_tab = self.findChild(QTabWidget, _name + 'results_tab')

        # Get the important widgets of the hys_results_tab
        choose_evalmodel = self.findChild(QComboBox, _name + 'choose_evalmodel')
        fit_report = results_tab.findChild(FitReportTable, _name + 'fit_report')
        results_report = results_tab.findChild(ResultsTable, _name + 'results_report')
        fit_points = results_tab.findChild(FitPointsTable, _name + 'fit_points_table')

        # Get the internal_id and the data package
        selected_widgets, selected_data = self.get_selection()
        _id, _pkg, _cols, _x = selected_data
        if _id is not None and _pkg:
            if _pkg == 'No data selected' or _pkg == 'raw_data':
                choose_evalmodel.clear()
                fit_report.clear()
                results_report.clear()
                fit_points.clear()
                return

            key = _pkg.split("_")[-1]
            results = getattr(data.dataset[_id], "results_" + key)

            # Check if the results are empty, i.e. {}
            if results == {}:
                choose_evalmodel.clear()
                fit_report.clear()
                results_report.clear()
                fit_points.clear()
                return

            else:
                choose_evalmodel.blockSignals(True)
                # Fill the QComboBox with the keys of the results
                old_evalmodel = choose_evalmodel.currentText()
                choose_evalmodel.clear()
                choose_evalmodel.addItems(list(results.keys()))
                if old_evalmodel in results.keys():
                    index = choose_evalmodel.findText(old_evalmodel)
                else:
                    index = 0
                choose_evalmodel.setCurrentIndex(index)
                # Calculate the width of the longest item
                longest_item = max(results.keys(), key=len)
                font_metrics = choose_evalmodel.fontMetrics()
                width = font_metrics.horizontalAdvance(longest_item) + 20  # Add some padding

                # Set the width of the QComboBox
                choose_evalmodel.setMinimumWidth(width)
                choose_evalmodel.blockSignals(False)
                
                # Process pending events
                QApplication.processEvents()

            if choose_evalmodel.currentText() in results.keys():
                # Fill the QTextEdit with the fit_report and results_report
                try:
                    fit_report.clear()
                    fit_report.fill_widget(results[choose_evalmodel.currentText()]["result"])
                except KeyError:
                    fit_report.clear()
                try:
                    results_report.clear()
                    results_report.fill_widget(results[choose_evalmodel.currentText()]["params"])
                except KeyError:
                    results_report.clear()

                try:
                    fit_points.clear()
                    fit_points.fill_widget(results[choose_evalmodel.currentText()]["fitted_data"])
                except KeyError:
                    fit_points.clear()
                    
                # Process pending events
                QApplication.processEvents()

    def update_fit_data(self, groupbox, evalmodel):
        # get the internal_id and the data package
        selected_widgets, selected_data = self.get_selection()
        _id, _pkg, _, _ = selected_data
        _, _, _, _, data_xdata = selected_widgets
        if _id is not None and _pkg:
            # Control the header of the (first) selected column
            if _pkg == 'No data selected':
                return
            
            # Create a new ExportTableWidget
            table_fitdata = ExportTableWidget()
            
            # get fitted data in data.dataset[_id].results[_pkg][evalmodel]
            fitted_data = getattr(data.dataset[_id].results[_pkg][evalmodel], 'fitted_data')
            
            # fill widget with fitted data
            horizontal_headers = list(fitted_data.keys())

            if isinstance(fitted_data, pd.DataFrame):
                # Set the number of rows and columns
                table_fitdata.setRowCount(fitted_data.shape[0] + 1)
                table_fitdata.setColumnCount(fitted_data.shape[1])

                # Add the attribute "unit" to the first row
                for j in range(fitted_data.shape[1]):
                    header = horizontal_headers[j]
                    # Attempt to get the unit attribute
                    if hasattr(fitted_data[header], 'unit'):
                        unit = getattr(fitted_data[header], 'unit', None)
                        quantity_type = None
                        for key in EvaluixConfig['conversion_factors']:
                            if str(unit) in EvaluixConfig['conversion_factors'][key].keys():
                                quantity_type = key
                                break
                        
                    else:
                        unit = 'unknown unit'  # or unit = ''
                        quantity_type = None
                    
                    # Create a QComboBox for unit selection
                    unit_selector = QComboBox()
                    if quantity_type is not None:
                        available_units = EvaluixConfig['conversion_factors'][quantity_type]
                    else:
                        available_units = [unit]
                    unit_selector.addItems(available_units)
                    unit_selector.setCurrentText(unit)
                    unit_selector.currentTextChanged.connect(lambda new_unit, header=header: 
                        self.update_unit(new_unit, 
                                         fitted_data,
                                         header, 
                                         table_fitdata,
                                         data_xdata))

                    table_fitdata.setCellWidget(0, j, unit_selector)

                for i in range(fitted_data.shape[0]):
                    for j in range(fitted_data.shape[1]):
                        # check if header is 'FlattendImage'
                        if horizontal_headers[j] == 'FlattendImage':
                            item = QTableWidgetItem("Image")
                        elif isinstance(fitted_data.iloc[i, j], (int, float)):
                            item = QTableWidgetItem("{:.5f}".format(fitted_data.iloc[i, j]))
                        else:
                            item = QTableWidgetItem(str(fitted_data.iloc[i, j]))
                        table_fitdata.setItem(i+1, j, item) # i is the row, j is the column

            # Set horizontal headers and update the table
            table_fitdata.setHorizontalHeaderLabels(horizontal_headers)
            
            vertical_headers = ['Unit'] + [str(i) for i in range(fitted_data.shape[0])]
            table_fitdata.setVerticalHeaderLabels(vertical_headers)

            if not table_fitdata.selectedIndexes() or len(table_fitdata.selectedIndexes()) != table_fitdata.rowCount() * len(set(index.column() for index in table_fitdata.selectedIndexes())):
                # Try to select column 1, if it exists
                if table_fitdata.columnCount() > 1:
                    table_fitdata.selectColumn(1)
                # If column 1 doesn't exist, select column 0
                elif table_fitdata.columnCount() > 0:
                    table_fitdata.selectColumn(0)

        else:
            # Add a dummy row of data to make the table visible and selectable
            table_fitdata.setRowCount(1)
            table_fitdata.setColumnCount(1)
            horizontal_headers = ['No data selected']
            item = QTableWidgetItem('Cols: 0, Rows: 0')
            table_fitdata.setItem(0, 0, item)
            # Set horizontal headers
            table_fitdata.setHorizontalHeaderLabels(horizontal_headers)

        # Update the table
        table_fitdata.resizeColumnsToContents()
        table_fitdata.update()

    def update_status_bar(self, message=None, saved_message=None):
        # Update the dataset label
        data_amount = len(data.dataset)
        self.status_dataset_label.setText(f"Loaded datasets: {data_amount} ")

        # show the Autosave_status from the ProfileConfig file
        if ProfileConfig['Autosave_status']:
            self.status_autosave_label.setText("Autosave: ON ")
            self.status_autosave_label.setStyleSheet("color: #ADD8E6")
            self.status_autosave_label.setToolTip(f"Autosave on filepath {self.save_path}")
            self.status_autosave_label.setToolTip(f""" <p style='background-color: rgb(45, 55, 65); border: 1px solid black;'>
        Autosave on filepath {self.save_path}
        </p>
        """)
        else:
            self.status_autosave_label.setText("Autosave: OFF ")
            self.status_autosave_label.setStyleSheet("color: yellow")
            self.status_autosave_label.setToolTip(f""" <p style='background-color: rgb(45, 55, 65); border: 1px solid black;'>
        Autosave is disabled
        </p>
        """)
            #self.status_autosave_label.setToolTip("Autosave is disabled")

        # Update the message label
        if message:
            message = message.strip() + " "
            self.status_message_label.setText(message)
        elif message == "" or message == 'clear':
            # If no message is provided, clear the message label
            self.message_label.clear()
        else:
            # If the message is None, do nothing beside checking if the current message "No data available. Please load a dataset."
            # is still valid which is not the case if data_amount > 0
            if self.status_message_label.text().strip() == "No data available. Please load a dataset." and data_amount > 0:
                self.status_message_label.clear()
            pass

        if saved_message:
            self.status_save_label.setText(saved_message)
        elif saved_message == "" or saved_message == 'clear':
            self.status_save_label.clear()
        else:
            pass

    def update_data_slider(self, data_slider):
        # get the internal_id and the data package
        selected_widgets, selected_data = self.get_selection()
        _id, _pkg, _cols, _x = selected_data
        if _id is not None and _pkg:
            if _pkg == 'No data selected':
                data_slider.clear()
                return

            else:
                _data = getattr(data.dataset[_id], _pkg)
                data_slider.setRange(0, _data.shape[0]-1)
                data_slider.setValue(0)

        else:
            data_slider.clear()
            data_slider.setRange(0, 0)
            data_slider.setValue(0)

    def update_sliderlabels(self):
        tab_main = self.findChild(QTabWidget, 'tab_main')

        # Get the currently open tab
        index = tab_main.currentIndex()
        tab_name = tab_main.widget(index).objectName()

        #identify tabs function by slicing the tab_name
        _name = tab_name.split("_")[-1] + "_" # the "_" is important due to current naming convention

        # get the field and img_nr labels
        field_label = self.findChild(QLabel, _name + 'field_label')
        img_nr_label = self.findChild(QLabel, _name + 'img_nr_label')

        # get the data_slider
        data_slider = self.findChild(JumpSlider, _name + 'data_slider')

        # put the values into the labels
        field_label.setText(f"Field: {data_slider.value()}")
        img_nr_label.setText(f"Image Nr: {data_slider.value()}")

    def slider_to_table(self, table, value):
        if not self.kerr_selection_changed:
            # Set the flag before changing the selection programmatically
            self.kerr_selection_changed = True
            table.selectRow(value)
            self.kerr_selection_changed = False

    def table_to_slider(self, slider, row):
        if not self.kerr_selection_changed:
            # Set the flag before changing the value programmatically
            self.kerr_selection_changed = True
            slider.setValue(row)
            self.kerr_selection_changed = False

    def update_all(self, type):
        tab_main = self.findChild(QTabWidget, 'tab_main')

        # Get the currently open tab
        index = tab_main.currentIndex()
        tab_name = tab_main.widget(index).objectName()

        #identify tabs function by slicing the tab_name
        _name = tab_name.split("_")[-1] + "_" # the "_" is important due to current naming convention
        # for example: hys_ or afm_
        
        # find the respective widgets
        table_metadata = self.findChild(DragDropTableWidget, _name + 'table_metadata')
        table_datapkg = self.findChild(QTableView, _name + 'table_datapkg')
        table_data = self.findChild(QTableView, _name + 'table_data')
        scroll_area = self.findChild(QScrollArea, _name + 'scroll_area')
        data_slider = self.findChild(JumpSlider, _name + 'data_slider')
        data_xdata = self.findChild(QComboBox, _name + 'data_xdata')
        loglist = self.findChild(QListWidget, _name + 'loglist')
        results_tab = self.findChild(QTabWidget, _name + 'results_tab')
        data_panel = self.findChild(QGroupBox, _name + 'data_panel')

        # block all connected signals to avoid update cascades
        # try/except because the following widgets are not always available
        try:
            table_metadata.blockSignals(True)
        except:
            pass
        try:
            table_datapkg.blockSignals(True)
        except:
            pass
        try:
            table_data.blockSignals(True)
        except:
            pass
        try:
            scroll_area.blockSignals(True)
        except:
            pass
        try:
            data_slider.blockSignals(True)
        except:
            pass
        try:
            data_xdata.blockSignals(True)
        except:
            pass

        # update the respective widgets
        #TODO: log the errors and find a better solution for this
        #print(f"An error occurred: {e}")
        try:
            self.update_table_metadata(table_metadata, type=type)
        except Exception as e:
            print(f"An error occurred: {e}." + traceback.format_exc())
            pass
        try:
            self.update_table_datapkg(table_datapkg)
        except Exception as e:
            pass
        try:
            self.update_table_data(table_data)
        except Exception as e:
            pass
        try:
            self.update_scroll_area(scroll_area) #TODO: implement this function
        except Exception as e:
            pass
        try:
            self.update_data_slider(data_slider)
        except Exception as e:
            pass
        try:
            self.update_choose_xdata(data_xdata)
        except Exception as e:
            pass
        try:
            self.update_loglist(loglist)
        except Exception as e:
            pass
        try:
            self.update_results_tab(results_tab)
        except Exception as e:
            pass
        try:
            if type == 'Hyst':
                self.visualize_selected_data(data_panel)
            elif type == 'AFM':
                self.visualize_selected_imgs(data_panel)
            elif type == 'Img':
                self.visualize_selected_imgs(data_panel)
        except Exception as e:
            pass
        try:
            self.update_status_bar(message=None)
        except Exception as e:
            pass

        # unblock all connected signals
        # try/except because the following widgets are not always available
        try:
            table_metadata.blockSignals(False)
        except:
            pass
        try:
            table_datapkg.blockSignals(False)
        except:
            pass
        try:
            table_data.blockSignals(False)
        except:
            pass
        try:
            scroll_area.blockSignals(False)
        except:
            pass
        try:
            data_slider.blockSignals(False)
        except:
            pass
        try:
            data_xdata.blockSignals(False)
        except:
            pass

    def update_all_tab(self):
        # This function just selects the correct type for the update_all function
        # depending on the current tab (order can be changed)

        index = self.tab_main.currentIndex()
        tab_name = self.tab_main.tabText(index)

        if tab_name == "Hysteresis":
            self.update_all(type='Hyst')
        elif tab_name == "AFM":
            self.update_all(type='Img')
        elif tab_name == "Kerr Microscope":
            self.update_all(type='Img')
        elif tab_name == "Datamanager":
            self.update_all(type='All')
        else:
            pass

    # 3. get, copy and delete selected data
    def get_selection(self):
        tab_main = self.findChild(QTabWidget, 'tab_main')

        # Get the currently open tab
        index = tab_main.currentIndex()
        tab_name = tab_main.widget(index).objectName()

        #identify tabs function by slicing the tab_name
        _name = tab_name.split("_")[-1] + "_" # the "_" is important due to current naming convention

        # find the respective widgets
        table_metadata = self.findChild(DragDropTableWidget, _name + 'table_metadata')
        table_datapkg = self.findChild(QTableView, _name + 'table_datapkg')
        table_data = self.findChild(QTableView, _name + 'table_data')
        data_xdata = self.findChild(QComboBox, _name + 'data_xdata')

        # initialize the variables and assign them to None
        _id = _pkg = _cols = _x = None

        # For each widget check if it exists and if it is selected. If this is correct, get the respective
        # selection and save it in the variable
        if table_metadata and table_metadata.selectionModel():
            internal_id = table_metadata.selectionModel().selectedRows()
            if internal_id:
                id_column = next((i for i in range(table_metadata.columnCount())
                                if table_metadata.horizontalHeaderItem(i).text() == 'ID'), None)
                if id_column is not None:
                    _id = int(table_metadata.item(internal_id[0].row(), id_column).text())

        if table_datapkg and table_datapkg.selectionModel():
            data_pkg = table_datapkg.selectionModel().selectedColumns()
            if data_pkg:
                _pkg = table_datapkg.model().headerData(data_pkg[0].column(), Qt.Orientation.Horizontal)

        if table_data and table_data.selectionModel():
            data_cols = table_data.selectionModel().selectedColumns()
            if data_cols:
                _cols = [table_data.model().headerData(col.column(), Qt.Orientation.Horizontal) for col in data_cols]

        if data_xdata and data_xdata.currentText() != "":
            _x = data_xdata.currentText()

        selected_widgets = tab_main, table_metadata, table_datapkg, table_data, data_xdata
        selected_data = _id, _pkg, _cols, _x

        return selected_widgets, selected_data

    def copy_dataset(self):
        # get the selected data
        selected_widgets, selected_data = self.get_selection()
        tab_main, table_metadata, table_datapkg, table_data, data_xdata = selected_widgets
        _id, _pkg, _cols, _x = selected_data

        # check if the selected data is valid
        if _id is not None and _pkg and _cols:
            if _id == 'No data selected':
                self.update_status_bar("No data selected. Please select a data package.")
                return
            else:
                # copy the data abd extend the dataset
                global data
                _metadata = copy.deepcopy(data.dataset[_id].metadata)
                _raw_data = copy.deepcopy(data.dataset[_id].raw_data)
                data.add_dataset(
                    Dataset(
                        _metadata,
                        _raw_data,
                    )
                )

                # get the new dataset (which has the highest internal_id)
                new_id = max(data.dataset.keys())

                used_keys = [int(key.split("_")[-1]) for key in data.dataset[_id].__dict__.keys() if "mod_data_" in key]
                # copy the mod_data_nr data packages
                for key in used_keys:
                    # check which mod_data_nr are available
                    if hasattr(data.dataset[_id], f'mod_data_{key}'):
                        # get the data
                        loglist = copy.deepcopy(getattr(data.dataset[_id], f'loglist_{key}'))
                        mod_data = copy.deepcopy(getattr(data.dataset[_id], f'mod_data_{key}'))
                        results = copy.deepcopy(getattr(data.dataset[_id], f'results_{key}'))

                        # add that the data are copied to the loglist
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        loglist.append(f"copied with Evaluix2_v{EvaluixConfig['Version']} @ {timestamp}")

                        #add the data to the new dataset
                        data.dataset[new_id].add_log_mod_results(key = key, loglist = loglist, mod_data = mod_data, results = results)

                        # select the new dataset
                        self.update_table_metadata(table_metadata)
                        # find the column index for 'ID'
                        id_column = -1
                        for i in range(table_metadata.columnCount()):
                            if table_metadata.horizontalHeaderItem(i).text() == 'ID':
                                id_column = i
                                break

                        # if 'ID' column is found
                        if id_column != -1:
                            for i in range(table_metadata.rowCount()):
                                if table_metadata.item(i, id_column).text() == str(new_key):
                                    table_metadata.selectRow(i)
                                    selected_index = table_metadata.selectedIndexes()[0]
                                    table_metadata.scrollToItem(table_metadata.item(i, selected_index.column()))
                                    break

                # update the status bar
                self.update_status_bar(f"Copied dataset {_id} to dataset {new_id}")

            if ProfileConfig['Autosave_status']:
                self.save_progress(ids=[_id])

    def del_dataset(self):
        selected_widgets, selected_data = self.get_selection()
        _id, _pkg, _cols, _x = selected_data

        if _id is not None:
            if _id == 'No data selected':
                self.update_status_bar("No data selected. Please select a data package.")
                return
            else:
                reply = QMessageBox.question(self, 'Delete Dataset',
                             f"Are you sure you want to delete dataset {_id}?",
                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                             QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    # first, change selected dataset to not cause display problems
                    tab_main = self.findChild(QTabWidget, 'tab_main')

                    # Get the currently open tab
                    index = tab_main.currentIndex()
                    tab_name = tab_main.widget(index).objectName()

                    #identify tabs function by slicing the tab_name
                    _name = tab_name.split("_")[-1] + "_" # the "_" is important due to current naming convention
                    table_metadata = self.findChild(QTableView, _name + 'table_metadata')
                    table_metadata.selectRow(0)

                    self.update_status_bar(f"Deleted dataset {_id}")
                    data.del_dataset(_id)

            if ProfileConfig['Autosave_status']:
                self.save_progress()

    def copy_mod_data(self):
        selected_widgets, selected_data = self.get_selection()
        tab_main, table_metadata, table_datapkg, table_data, data_xdata = selected_widgets
        _id, _pkg, _cols, _x = selected_data

        if _id is not None and _pkg and _cols:
            if _pkg == 'No data selected':
                self.update_status_bar("No data selected. Please select a data package.")
                return

            elif "mod_data_" in _pkg:
                # extract the current mod_data_nr
                key = _pkg.split("_")[-1]
                loglist = copy.deepcopy(getattr(data.dataset[_id], "loglist_" + key))
                mod_data = copy.deepcopy(getattr(data.dataset[_id], _pkg))
                results = copy.deepcopy(getattr(data.dataset[_id], "results_" + key))

            else:
                key = 0
                loglist = []
                mod_data = copy.deepcopy(getattr(data.dataset[_id], _pkg))
                results = {}

            # add that the data are copied to the loglist
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # get the current timestamp
            loglist.append(f"copied with Evaluix2_v{EvaluixConfig['Version']} @ {timestamp}")

            # get the new mod_data_nr which should be the highest number + 1 or larger
            used_keys = [int(key.split("_")[-1]) for key in data.dataset[_id].__dict__.keys() if "mod_data_" in key]
            key = int(key)

            if len(used_keys) < 100:
                new_key = key + 1
                while new_key in used_keys and new_key < 100:
                    new_key += 1

                if new_key < 100:
                    # add the data to the new dataset
                    data.dataset[_id].add_log_mod_results(key = new_key, loglist = loglist, mod_data = mod_data, results = results)
                    self.update_status_bar(f"Copied data package {key} to data package {new_key} of dataset {_id}")

                    # select the new data package
                    self.update_table_datapkg(table_datapkg)
                    header = table_datapkg.horizontalHeader()
                    for i in range(header.count()):
                        if table_datapkg.horizontalHeaderItem(i).text() == 'mod_data_' + str(new_key):
                            table_datapkg.selectColumn(i)
                            selected_index = table_datapkg.selectedIndexes()[0]
                            table_datapkg.scrollToItem(table_datapkg.item(selected_index.row(), 0))
                            break

                    if ProfileConfig['Autosave_status']:
                        self.save_progress(ids=[_id])

                    return new_key

                elif key > 0:
                    new_key = key - 1
                    while new_key in used_keys and new_key > 0:
                        new_key -= 1

                    if new_key > 0:
                        # add the data to the new dataset
                        data.dataset[_id].add_log_mod_results(key = new_key, loglist = loglist, mod_data = mod_data, results = results)
                        self.update_status_bar(f"Copied data package {key} to data package {new_key} of dataset {_id}")

                        # select the new data package
                        self.update_table_datapkg(table_datapkg)
                        header = table_datapkg.horizontalHeader()
                        for i in range(header.count()):
                            if table_datapkg.horizontalHeaderItem(i).text() == 'mod_data_' + str(new_key):
                                table_datapkg.selectColumn(i)
                                selected_index = table_datapkg.selectedIndexes()[0]
                                table_datapkg.scrollToItem(table_datapkg.item(selected_index.row(), 0))
                                break

                        if ProfileConfig['Autosave_status']:
                            self.save_progress(ids=[_id])

                        return new_key

                else:
                    self.update_status_bar("No new data package could be created. Please delete some old data packages.")
                    return
            else:
                self.update_status_bar("No new data package could be created. Please delete some old data packages.")
                return

    def del_mod_data(self):
        selected_widgets, selected_data = self.get_selection()
        _id, _pkg, _cols, _x = selected_data

        if _id is not None and _pkg and _cols:
            if _pkg == 'No data selected':
                self.update_status_bar("No data selected. Please select a data package.")
                return
            elif _pkg == 'raw_data':
                self.update_status_bar("You can't delete the raw_data. Instead you have to delete the whole dataset.")
                return
            else:
                reply = QMessageBox.question(self, 'Delete Data Package',
                                         f"Are you sure you want to delete data package {_pkg} of dataset {_id}?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    # first, change selected data package to "raw_data" as new displayed data package
                    tab_main = self.findChild(QTabWidget, 'tab_main')

                    # Get the currently open tab
                    index = tab_main.currentIndex()
                    tab_name = tab_main.widget(index).objectName()

                    #identify tabs function by slicing the tab_name
                    _name = tab_name.split("_")[-1] + "_" # the "_" is important due to current naming convention
                    table_datapkg = self.findChild(QTableView, _name + 'table_datapkg')
                    table_datapkg.selectColumn(0)

                    key = _pkg.split("_")[-1]
                    self.update_status_bar(f"Deleted data package {key} of dataset {_id}")
                    data.dataset[_id].del_log_mod_results(key)

            if ProfileConfig['Autosave_status']:
                self.save_progress(ids=[_id])

    def copy_datacol(self):
        selected_widgets, selected_data = self.get_selection()
        tab_main, table_metadata, table_datapkg, table_data, data_xdata = selected_widgets
        _id, _pkg, _cols, _x = selected_data

        if _id is not None and _pkg and _cols:
            if _pkg == 'No data selected':
                self.update_status_bar("No data selected. Please select a data package.")
                return

            # extract the data
            _data = getattr(data.dataset[_id], _pkg)
            if isinstance(_data, pd.DataFrame):
                # get the selected column
                col = copy.deepcopy(_data[_cols[0]])

                _data[_cols[0] + "_copy"] = col.copy()
                self.update_status_bar(f"Copied column {_cols[0]} to column {_cols[0] + '_copy'} of data package {_pkg} of dataset {_id}")

                # select the new data column
                self.update_table_data(table_data)
                header = table_data.horizontalHeader()
                for i in range(header.count()):
                    print(table_data.horizontalHeaderItem(i).text())
                    if table_data.horizontalHeaderItem(i).text() == _cols[0] + "_copy":
                        table_data.selectColumn(i)
                        selected_index = table_data.selectedIndexes()[0]
                        table_data.scrollToItem(table_data.item(selected_index.row(), 0))
                        break


            else:
                self.update_status_bar("No data selected. Please select a data package.")

            if ProfileConfig['Autosave_status']:
                self.save_progress(ids=[_id])

    def change_metadata(self, item):
        # get the current column
        column = item.column()

        # get the horizontal header of the current column
        metadata_key = self.dm_table_metadata.horizontalHeaderItem(column).text()
        
        if metadata_key == "ID":
            self.update_status_bar(message = "You can't change the ID of a dataset.")
            return

        # find the index of the "ID" column
        id_column = None
        for i in range(self.dm_table_metadata.columnCount()):
            if self.dm_table_metadata.horizontalHeaderItem(i).text() == "ID":
                id_column = i
                break

        # get the item in the "ID" column of the current row
        id_item = self.dm_table_metadata.item(item.row(), id_column) if id_column is not None else None

        # get the text of the id_item
        _id = int(id_item.text()) if id_item else None
        
        # get the current value and its type
        current_value = data.dataset[_id].metadata[metadata_key]
        current_type = type(current_value)

        # try to convert the new value to the current type
        try:
            new_value = current_type(item.text())
        except ValueError:
            raise ValueError(f"Cannot convert {item.text()} to {current_type}")
        
        self.update_status_bar(message = f"Changed metadata {metadata_key} of dataset {_id} to {new_value}")
        # set the new value
        data.dataset[_id].metadata[metadata_key] = new_value

    def add_data_manually(self):
        global data
        _id = data.id + 1
        while _id in data.dataset.keys():
            _id =+1
            if _id > 1000:
                raise ValueError("Too many datasets in the global dataset. No free ID available.")
        self.manual_data_input =  ManualDataDialog(self, _id)
        
        self.manual_data_input.applied.connect(
            lambda metadata, raw_data: 
            data.add_dataset(Dataset(
                metadata,
                pd.DataFrame(raw_data),
                )
            )
        )
        

    # Modify the keyPressEvent function to delete data packages with the delete key
    def keyPressEvent(self, event):
        if hasattr(event, 'key') and event.key() == Qt.Key.Key_Delete: # check if the key is actually a key and is the delete key

            self.map_groupbox_to_delbutton = {
                'hys_metadata_panel': self.hys_del_dataset,
                'hys_data_packages': self.hys_del_mod_data,
                'kerr_metadata_panel': self.kerr_del_dataset,
                'kerr_data_packages': self.kerr_del_mod_data,
                'afm_metadata_panel': self.afm_del_dataset,
                'afm_data_packages': self.afm_del_mod_data,
            }

            active_widget = QApplication.instance().focusWidget()
            while active_widget:
                if isinstance(active_widget, QGroupBox) and active_widget.objectName() in self.map_groupbox_to_delbutton:
                    button = self.map_groupbox_to_delbutton[active_widget.objectName()]
                    if hasattr(button, 'click'):
                        button.click()
                    break
                active_widget = active_widget.parent()

    # 4. Data visualization, manipulation and evaluation
    def visualize_selected_data(self, qgroupbox):
        # Get the selected dataset, data package, data columns and the chosen xdata
        selected_widgets, selected_data = self.get_selection()
        _id, _pkg, _cols, _x = selected_data

        # clear the canvas with the function clear() which is defined in the MyMplCanvas class
        qgroupbox.canvas.clear()

        if _id is not None and _pkg and _cols:
            try:
                _x_unitless = _x.split("[")[0].strip()
            except AttributeError:
                _x_unitless = _x
            plottable_data = [[_id, _pkg, _cols, _x_unitless]]

            # get the content of the table in which the preserved datasets are saved
            row_count = qgroupbox.table.rowCount()
            column_count = qgroupbox.table.columnCount()
            table_content = [[qgroupbox.table.item(i, j).text() for j in range(column_count)] for i in range(row_count)]

            # check if there are datasets in the table which have fitting data
            if table_content:
                for preserved_id, preserved_pkg in table_content:
                    preserved_id = int(preserved_id)
                    if not _id == preserved_id:#TODO: delete: and not _pkg == preserved_pkg:
                        if hasattr(data.dataset[preserved_id], preserved_pkg):
                            print(f"Preserved data package {preserved_pkg} of dataset {preserved_id} exists.")
                            # first check if the preserved data has a column fulfilling the requirement of _x
                            if not _x_unitless in getattr(data.dataset[preserved_id], preserved_pkg).columns and not _x_unitless == "Point number":
                                print(f"Preserved data package {preserved_pkg} of dataset {preserved_id} does not have a column {_x_unitless}.")
                                # if not, take next preserved dataset
                                continue

                            for col in getattr(data.dataset[preserved_id], preserved_pkg).columns:
                                print(f"Preserved data package {preserved_pkg} of dataset {preserved_id} has column {col}.")
                                if col in _cols:
                                    # preserve the data a second time to plot it in the same plot
                                    plottable_data.append([preserved_id, preserved_pkg, [col], _x_unitless])
                                    print([preserved_id, preserved_pkg, [col], _x_unitless])
                        else:
                            self.update_status_bar(f"Preserved data package {preserved_pkg} of dataset {preserved_id} does not exist.")

            x_min = x_max = y_min = y_max = None
            for _id, _pkg, _cols, _x_unitless in plottable_data:
                try:
                    try:
                        if _x_unitless == "Point number":
                            xdata = np.arange(len(getattr(data.dataset[_id], _pkg)[_cols[0]]))
                        else:
                            xdata = getattr(data.dataset[_id], _pkg)[_x_unitless]
                    except KeyError:
                        xdata = getattr(data.dataset[_id], _pkg).iloc[:, 0]

                    if x_min is None or np.min(xdata) < x_min:
                        x_min = np.min(xdata)
                    if x_max is None or np.max(xdata) > x_max:
                        x_max = np.max(xdata)

                    for col in _cols:
                        # Get the data
                        _data = getattr(data.dataset[_id], _pkg)[col]
                        # Check if qgroupbox.canvas.purpose is 'Hysteresis'
                        if qgroupbox.canvas.purpose == 'Hysteresis': # Try to plot both branches in slightly different colors given by qgroupbox.canvas prop_cycler
                            try:
                                # Get the current color from the color cycle
                                current_color = qgroupbox.canvas.axes._get_lines.get_next_color()
                                # Find the corresponding darker color from the color_list
                                darker_color = None
                                for color_pair in qgroupbox.canvas.color_list:
                                    if color_pair[0].lower() == current_color.lower():
                                        darker_color = color_pair[1]
                                        break
                            
                                # Plot the second branch with the corresponding darker color
                                if darker_color:
                                    # Slice xdata and ydata into two equally sized arrays, i.e. the branches
                                    xdata1, xdata2 = np.array_split(xdata, 2)
                                    _data1, _data2 = np.array_split(_data, 2)
                                    
                                    # Plot the first branch
                                    qgroupbox.canvas.axes.plot(xdata1, _data1, label=col, zorder=1, marker='o', markersize=3, lw=0.5, color=current_color)
                                    # Plot the second branch
                                    qgroupbox.canvas.axes.plot(xdata2, _data2, zorder=1, marker='o', markersize=3, lw=0.5, color=darker_color)
                                
                            except Exception as e:
                                print(f"An error occurred while plotting the hysteresis data: {e}")
                                # Just plot the whole hysteresis loop in one color
                                qgroupbox.canvas.axes.plot(xdata, _data, label=col, zorder=1, marker='o', markersize=3, lw=0.5)
                                    
                        else:
                            # Plot the data
                            qgroupbox.canvas.axes.plot(xdata, _data, label=col, zorder=1, marker='o', markersize=3, lw=0.5)
                        # Set the labels
                        try:
                            #first argument is the x-axis label, second argument is the y-axis label
                            qgroupbox.canvas.set_labels(xdata.name + " [" + xdata.unit + "]", _data.name + " [" + _data.unit + "]")
                        except AttributeError:
                            try:
                                qgroupbox.canvas.set_labels(_x, _data.name + " [" + _data.unit + "]")
                            except AttributeError:
                                qgroupbox.canvas.set_labels(_x, _data.name)

                        if y_min is None or np.min(_data) < y_min:
                            y_min = np.min(_data)
                        if y_max is None or np.max(_data) > y_max:
                            y_max = np.max(_data)
                except Exception as e:
                    self.update_status_bar(f"An error occurred while plotting the data: {e}")
                    print(f"An error occurred while plotting the data: {e}")

            # Automatically scale the view to the data
            if x_min is not None and x_max is not None and y_min is not None and y_max is not None:
                x_range = x_max - x_min
                y_range = y_max - y_min
                qgroupbox.canvas.axes.set_xlim([x_min - 0.05*x_range, x_max + 0.05*x_range])
                qgroupbox.canvas.axes.set_ylim([y_min - 0.05*y_range, y_max + 0.05*y_range])
            else:
                qgroupbox.canvas.axes.autoscale()

            qgroupbox.canvas.axes.legend()
            qgroupbox.canvas.figure.tight_layout()
            qgroupbox.canvas.draw()

    def preserve_current_datapgk(self, qgroupbox):
        # Get the selected dataset, data package, data columns and the chosen xdata
        selected_widgets, selected_data = self.get_selection()
        _id, _pkg, _cols, _x = selected_data

        # check if _id and _pkg are valid
        if _id is not None and _pkg and _cols:
            # get the QTableWidget (qgroupbox.table) which contains the plotted data and add the data to the table if it is not already there

            # get the HorizontalHeaderLabels
            horizontal_headers = [qgroupbox.table.horizontalHeaderItem(i).text() for i in range(qgroupbox.table.columnCount())]
            # check if table has been initialized
            if not horizontal_headers == ["Dataset id", "Data Package"]:
                horizontal_headers = ["Dataset id", "Data Package"]
                # set the horizontal headers
                qgroupbox.table.setColumnCount(len(horizontal_headers))
                qgroupbox.table.setHorizontalHeaderLabels(horizontal_headers)

            # get the number of rows and the content of the table
            row_count = qgroupbox.table.rowCount()
            column_count = qgroupbox.table.columnCount()
            table_content = [[qgroupbox.table.item(i, j).text() for j in range(column_count)] for i in range(row_count)]

            # check if the data is already in the table
            if [str(_id), _pkg] not in table_content:
                # add the data to the table
                qgroupbox.table.setRowCount(row_count + 1)
                for j, data in enumerate([_id, _pkg]):
                    item = QTableWidgetItem(str(data))
                    qgroupbox.table.setItem(row_count, j, item)
                qgroupbox.table.resizeColumnsToContents()
                qgroupbox.table.update()

            else:
                self.update_status_bar("The selected data package is already in the table.")

        else:
            self.update_status_bar("Invalid data selected. Please select a dataset and a data package.")

    def del_preserved_content(self, qgroupbox):
        # get the selection in the qgroupbox.table
        selection = qgroupbox.table.selectionModel().selectedRows()

        if selection:
            # get the row numbers in reverse order to not mess up the indices
            rows = sorted([index.row() for index in selection], reverse=True)

            # remove the rows from the table
            for row in rows:
                qgroupbox.table.removeRow(row)

            qgroupbox.table.resizeColumnsToContents()
            qgroupbox.table.update()

    def visualize_selected_imgs(self, qgroupbox):
        # Get the selected dataset, data package, data columns and the chosen xdata
        selected_widgets, selected_data = self.get_selection()
        tab_main, table_metadata, table_datapkg, table_data, data_xdata = selected_widgets
        _id, _pkg, _cols, _x = selected_data

        # # Check if the QGroupbox has a Qslider which is used to select the image to be plotted
        # data_slider = qgroupbox.findChild(JumpSlider, 'kerr_data_slider')
        # if data_slider is not None:
        #     # if yes, get the slider value which is the image number to be plotted
        #     try:
        #         img_nr = qgroupbox.kerr_data_slider.value()
        #     except:
        #         img_nr = 0

        # clear the canvas with the function clear() which is defined in the MyMplCanvas class
        qgroupbox.canvas.clear()
        
        if _id is not None and _pkg and table_data:
            
            # in this case, the rows of the table_data is important, not the cols
            # get the content of the table_data
            if table_data and table_data.selectionModel():
                data_rows = table_data.selectionModel().selectedRows()
            
            if data_rows:
                # get the row number of the selected data
                row = data_rows[0].row()
                # get the selected data
                flattendimg = np.asarray(getattr(data.dataset[_id], _pkg)['FlattendImage'][row])
                imgshape = getattr(data.dataset[_id], _pkg)['Shape (px)'][row]

                img = flattendimg.reshape(imgshape)
                qgroupbox.canvas.axes.imshow(img, cmap='gray')

                qgroupbox.canvas.figure.tight_layout()
                qgroupbox.canvas.draw()
            
            # # if img_nr is not defined, set it to 0
            # if not 'img_nr' in locals(): #TODO: work on this
            #     img_nr = 0

            # # for now as a test, just plot the first image of the dataset
            # flattendimg = np.asarray(getattr(data.dataset[_id], _pkg)['FlattendImage'][img_nr])
            # imgshape = getattr(data.dataset[_id], _pkg)['Shape (px)'][img_nr]

            # img = flattendimg.reshape(imgshape)
            # qgroupbox.canvas.axes.imshow(img, cmap='gray')

            # qgroupbox.canvas.figure.tight_layout()
            # qgroupbox.canvas.draw()

    def data_manipulation(self, func, add_xdata=False, **kwargs):
        # Get the selected dataset, data package, data columns and the chosen xdata
        selected_widgets, selected_data = self.get_selection()
        tab_main, table_metadata, table_datapkg, table_data, data_xdata = selected_widgets
        _id, _pkg, _cols, _x = selected_data

        if _id is not None and _pkg and _cols:

            # first check if new data has to be created due to
            # 1. the data package is "raw_data"
            # 2. the data package is not "raw_data" but the new_data_flag is checked
            # 3. the function requires another data length than the original data (currently only num_derivative)
            tab_main = self.findChild(QTabWidget, 'tab_main')
            # Get the currently open tab
            index = tab_main.currentIndex()
            tab_name = tab_main.widget(index).objectName()
            #identify tabs function by slicing the tab_name
            _name = tab_name.split("_")[-1] + "_" # the "_" is important due to current naming convention
            new_data_flag = self.findChild(QCheckBox, _name + 'new_data_flag')

            # functions which require a new data package
            funcs_with_new_data = [num_derivative, num_integral]

            if _pkg == "raw_data" or new_data_flag.isChecked() or func in funcs_with_new_data:
                key = self.copy_mod_data()
                _pkg = f"mod_data_{key}"

            for col in _cols:
                # Get the data
                _data = getattr(data.dataset[_id], _pkg)[col]
                
                # save the unit if there is any
                unit = _data.unit if hasattr(_data, 'unit') else None
                print(f"unit: {unit}")

                # extract the keyword arguments and their values from the yaml file as input for the function
                kwargs_yaml = copy.deepcopy(ProfileConfig['function_info'][func.__name__])

                # check if additional keyword arguments are given and
                if kwargs:
                    for key, value in kwargs.items():
                        kwargs_yaml[key]['value'] = value

                # create a dictionary with the keyword arguments and their default/input values
                type_mapping = {
                    'int': int,
                    'float': float,
                    'str': str,
                    'list': list,
                    'dict': dict,
                    'Union': Union,
                    'pd.DataFrame': pd.DataFrame,
                    'pd.Series': pd.Series,
                    'np.ndarray': np.ndarray,
                    'bool': str_to_bool,
                    # Add more types here if necessary
                }

                kwargs = {} # new dictionary to store the keyword arguments and their values
                for key in kwargs_yaml.keys():
                    if key in ['xdata', 'ydata', 'Category']:
                        continue

                    expected_type_str = ProfileConfig['function_info'][func.__name__][key]['type']
                    value = ProfileConfig['function_info'][func.__name__][key]['value']
                    if 'Union' in expected_type_str:
                        expected_types = [type_mapping[type_str.strip()] for type_str in expected_type_str.replace('Union[', '').replace(']', '').split(',')]
                        for expected_type in expected_types:
                            try:
                                kwargs[key] = expected_type(value)
                                break
                            except ValueError:
                                continue
                        else:
                            self.update_status_bar(f"The input {value} is not valid for the type {expected_type_str}.")
                            return
                    else:
                        expected_type = type_mapping[expected_type_str]
                        try:
                            kwargs[key] = expected_type(value)
                        except ValueError:
                            self.update_status_bar(f"The input {value} is not valid for the type {expected_type_str}.")
                            return


                # kwargs = {key: type_mapping[kwargs_yaml[key]['type']](kwargs_yaml[key]['default']) for key in kwargs_yaml.keys()}

                if add_xdata:
                    # Get the x-data
                    if _x == "Point number":
                        xdata = np.arange(len(_data))
                    else:
                        try:
                            xdata = getattr(data.dataset[_id], _pkg)[_x]
                        except KeyError:
                            xdata = getattr(data.dataset[_id], _pkg).iloc[:, 0]
                            self.update_status_bar("KeyError: The selected xdata is not in the data package. Using the first column instead.")

                    unitx = xdata.unit if hasattr(xdata, 'unit') else None
                    
                    # Manipulate the data
                    print(kwargs)
                    new_data = func(xdata, _data, **kwargs)
                else:
                    new_data = func(_data, **kwargs)

                if func in funcs_with_new_data:
                    # Create a dictionary where the keys are the column names and the values are the columns of new_data
                    data_dict = {f"{col}_{i}": new_data[i] for i in range(len(new_data))}
                    # Overwrite the data
                    try:
                        key = _pkg.split("_")[-1]
                        data.dataset[_id].update_mod_data(key = key, new_data = data_dict)
                        for col_key in data_dict.keys():
                            if col_endswith('_0'):
                                getattr(data.dataset[_id], _pkg)[col_key].unit = unitx
                            else:
                                getattr(data.dataset[_id], _pkg)[col_key].unit = unit
                        print('new data set')
                    except Exception as e:
                        print(f"An error occurred while setting the new data: {e}")
                        
                else:
                    # Overwrite the data
                    try:
                        if len(new_data) == 2 and add_xdata:
                            getattr(data.dataset[_id], _pkg)[col] = new_data[1]
                            getattr(data.dataset[_id], _pkg)[_x] = new_data[0] if not _x == "Point number" else np.arange(len(new_data[1]))
                            if unit:
                                if hasattr(getattr(data.dataset[_id], _pkg)[col], 'unit'):
                                    getattr(data.dataset[_id], _pkg)[col].unit = unit
                                if hasattr(getattr(data.dataset[_id], _pkg)[_x], 'unit'):
                                    getattr(data.dataset[_id], _pkg)[_x].unit = unitx
                        else:
                            getattr(data.dataset[_id], _pkg)[col] = new_data
                            if unit and hasattr(getattr(data.dataset[_id], _pkg)[col], 'unit'):
                                getattr(data.dataset[_id], _pkg)[col].unit = unit

                        print('new data set')
                    except Exception as e:
                        print('failed to set new data')
                        print(f"An error occurred while setting the new data: {e}")
                    

                # update the loglist with the function name and the arguments
                log_func = f"data manipulation via {func.__name__} at {col}"
                log_xdata = f" with xdata = {_x}" if add_xdata else ""
                if kwargs:
                    log_kwargs = ' with kwargs: ' + ', '.join(f"{k}={v}" for k, v in kwargs.items())
                else:
                    log_kwargs = ""

                logmessage = log_func + log_xdata + log_kwargs

                loglist = getattr(data.dataset[_id], "loglist_" + _pkg.split("_")[-1])
                loglist.append(logmessage)

            if ProfileConfig['Autosave_status']:
                self.save_progress(ids=[_id])

    def data_evaluation(self, func, add_xdata=False, **kwargs):
        #TODO: how to proceed with the results? They should be stored within the
        # results of the data package which is a dictionary. Name of the result should be the column name + the function name + fit if it is a fit.

        selected_widgets, selected_data = self.get_selection()
        _id, _pkg, _cols, _x = selected_data

        if _id is not None and _pkg and _cols:
            # first check if new data has to be created due to _pkg being "raw_data"
            if _pkg == "raw_data":
                key = self.copy_mod_data()
                _pkg = f"mod_data_{key}"

            for col in _cols:
                # Get the data
                _data = getattr(data.dataset[_id], _pkg)[col]

                # extract the keyword arguments and their values from the yaml file as input for the function
                kwargs_yaml = copy.deepcopy(ProfileConfig['function_info'][func.__name__])

                # check if additional keyword arguments are given and
                if kwargs:
                    for key, value in kwargs.items():
                        kwargs_yaml[key]['value'] = value

                # create a dictionary with the keyword arguments and their default/input values
                type_mapping = {
                    'int': int,
                    'float': float,
                    'str': str,
                    'list': list,
                    'dict': dict,
                    'Union': Union,
                    'pd.DataFrame': pd.DataFrame,
                    'pd.Series': pd.Series,
                    'np.ndarray': np.ndarray,
                    'bool': str_to_bool,
                    # Add more types here if necessary
                }

                kwargs = {} # new dictionary to store the keyword arguments and their values
                for key in kwargs_yaml.keys():
                    if key in ['xdata', 'ydata', 'Category']:
                        continue

                    expected_type_str = ProfileConfig['function_info'][func.__name__][key]['type']
                    value = ProfileConfig['function_info'][func.__name__][key]['value']
                    if 'Union' in expected_type_str:
                        expected_types = [type_mapping[type_str.strip()] for type_str in expected_type_str.replace('Union[', '').replace(']', '').split(',')]
                        for expected_type in expected_types:
                            try:
                                kwargs[key] = expected_type(value)
                                break
                            except ValueError:
                                continue
                        else:
                            self.update_status_bar(f"The input {value} is not valid for the type {expected_type_str}.")
                            return
                    else:
                        expected_type = type_mapping[expected_type_str]
                        try:
                            kwargs[key] = expected_type(value)
                        except ValueError:
                            self.update_status_bar(f"The input {value} is not valid for the type {expected_type_str}.")
                            return

                if add_xdata:
                    # Get the x-data
                    if _x == "Point number":
                        xdata = np.arange(len(_data))
                    else:
                        try:
                            xdata = getattr(data.dataset[_id], _pkg)[_x]
                        except KeyError:
                            xdata = getattr(data.dataset[_id], _pkg).iloc[:, 0]
                            self.update_status_bar("KeyError: The selected xdata is not in the data package. Using the first column instead.")

                    # Evaluate the data
                    eval_result = func(xdata, _data, **kwargs)
                else:
                    eval_result = func(_data, **kwargs)

                if len(eval_result) == 3:
                    fitted_data, params, result = eval_result
                    result_dict = {
                        'function': func.__name__,
                        'fitted_data': fitted_data,
                        'params': params, 
                        'result': result}
                elif len(eval_result) == 2:
                    fitted_data, result = eval_result
                    result_dict = {
                        'function': func.__name__,
                        'fitted_data': fitted_data,
                        'result': result}
                else:
                    params = eval_result
                    result_dict = {
                        'function': func.__name__,
                        'params': params}
                    
                # update the loglist with the function name and the arguments
                log_func = f"data evaluation via {func.__name__} at {col} "
                log_xdata = f" with xdata = {_x}" if add_xdata else ""
                if kwargs:
                    log_kwargs = 'with kwargs: ' + ', '.join(f"{k}={v}" for k, v in kwargs.items())
                else:
                    log_kwargs = ""

                logmessage = log_func + log_xdata + log_kwargs

                loglist = getattr(data.dataset[_id], "loglist_" + _pkg.split("_")[-1])
                loglist_pos = len(loglist)
                loglist.append(logmessage)
                result_dict['loglist_pos'] = loglist_pos

                results = getattr(data.dataset[_id], f"results_{_pkg.split('_')[-1]}")
                results_key = f'{func.__name__}' + f'_at_log_{loglist_pos}'
                results[results_key] = result_dict

            if ProfileConfig['Autosave_status']:
                self.save_progress(ids=[_id])

        self.update_status_bar(f"Data evaluation via {func.__name__} completed.")
        self.update_results_tab()

    # 5. Info, settings and configuration
    def info_settings(self, func, profile=False):
        if not profile:
            # Create a new instance of InfoSettingsDialog
            self.dialog = InfoSettingsDialog()
            self.dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

            ############ Info Tab ############
            # Find the QLabel in the Info tab
            self.info_funclabel = self.dialog.findChild(QLabel, 'info_funclabel')
            # Set the text of the QLabel to the function name
            func_name = f"Function name: {func.__name__}"
            self.info_funclabel.setText(func_name)
            # Find the QTextEdit in the Info tab
            self.info_text = self.dialog.findChild(QTextEdit, 'info_text')
            # Set the text of the QLabel to the docstring
            docstring = inspect.getdoc(func)
            self.info_text.setPlainText(docstring)

            ############ Settings Tab ############
            # Find the QLabel in the Settings tab
            self.settings_funclabel = self.dialog.findChild(QLabel, 'settings_funclabel')
            # Set the text of the QLabel to the function name
            self.settings_funclabel.setText(func_name)
            # Find the QTableWidget in the Settings tab
            self.settings_table = self.dialog.findChild(QTableWidget, 'settings_table')
            # Get the signature of the function
            parameters = {name: [param['type'], param['value'], param['default']] for name, param in ProfileConfig['function_info'][func.__name__].items() if name != 'Category'}
            parameters['Category'] = ['str', 'This settles the GUI logic.\nDo not change.', ProfileConfig['function_info'][func.__name__]['Category']]
            # Set the number of rows in the table to the number of parameters
            self.settings_table.setRowCount(len(parameters))
            # Set the number of columns in the table to 3 (for name, type, value and default value)
            self.settings_table.setColumnCount(4)
            # Set the headers of the table
            self.settings_table.setHorizontalHeaderLabels(['Name', 'Type', 'Value', 'Default'])
            # Populate the table with the parameters
            for i, (name, param) in enumerate(parameters.items()):
                # Create QTableWidgetItem objects for the name, type, and value
                name_item = QTableWidgetItem(name)
                if name_item.text() in ['xdata', 'ydata']:
                    type_item = QTableWidgetItem('pd.DataFrame')
                    value_item = QTableWidgetItem('Select in data table')
                    default_item = QTableWidgetItem('-')
                else:
                    type_item = QTableWidgetItem(str(param[0]))
                    value_item = QTableWidgetItem(str(param[1]) if param[1] != '-' else 'No value given')
                    default_item = QTableWidgetItem(str(param[2]) if param[2] != '-' else 'No default')

                # Make the name, type, and default items not editable
                name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                default_item.setFlags(default_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

                # Add the items to the table
                self.settings_table.setItem(i, 0, name_item)
                self.settings_table.setItem(i, 1, type_item)
                self.settings_table.setItem(i, 2, value_item)
                self.settings_table.setItem(i, 3, default_item)

            # Resize the columns to fit their contents
            self.settings_table.resizeColumnsToContents()

            # connect the settings_table to the update_yaml function
            self.settings_table.itemChanged.connect(lambda item: self.safe_execute(lambda: self.update_yaml(item, func=func)))

            ############ Display InfoSett ############
            self.dialog.show()
            
        else:
            # No new dialog or Labels needed, just filling the docstring and the settings table
            # Find the QTextEdit in the Info tab and fill it with the docstring
            self.info_text = self.dialog.findChild(QTextEdit, 'info_text')
            docstring = inspect.getdoc(func)
            self.info_text.setPlainText(docstring)
            
            # Find the QTableWidget in the Settings tab and fill it with the ProfileConfig params
            self.funcsettings_table = self.dialog.findChild(QTableWidget, 'funcsettings_table')
            # Get the signature of the function in the ProfileConfig
            parameters = {name: [param['type'], param['value'], param['default']] for name, param in ProfileConfig['function_info'][func].items() if name != 'Category'}
            parameters['Category'] = ['str', 'This settles the GUI logic.\nDo not change.', ProfileConfig['function_info'][func]['Category']]
            # Set the number of rows in the table to the number of parameters
            self.funcsettings_table.setRowCount(len(parameters))
            # Set the number of columns in the table to 4 (for name, type, value and default value)
            self.funcsettings_table.setColumnCount(4)
            # Set the headers of the table
            self.funcsettings_table.setHorizontalHeaderLabels(['Name', 'Type', 'Value', 'Default'])
            # Populate the table with the parameters
            for i, (name, param) in enumerate(parameters.items()):
                # Create QTableWidgetItem objects for the name, type, and value
                name_item = QTableWidgetItem(name)
                if name_item.text() in ['xdata', 'ydata']:
                    type_item = QTableWidgetItem('pd.DataFrame')
                    value_item = QTableWidgetItem('Select in data table')
                    default_item = QTableWidgetItem('-')
                else:
                    type_item = QTableWidgetItem(str(param[0]))
                    value_item = QTableWidgetItem(str(param[1]) if param[1] != '-' else 'No value given')
                    default_item = QTableWidgetItem(str(param[2]) if param[2] != '-' else 'No default')

                # Make the name, type, and default items not editable
                name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                default_item.setFlags(default_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

                # Add the items to the table
                self.funcsettings_table.setItem(i, 0, name_item)
                self.funcsettings_table.setItem(i, 1, type_item)
                self.funcsettings_table.setItem(i, 2, value_item)
                self.funcsettings_table.setItem(i, 3, default_item)
            
            # Resize the columns to fit their contents
            self.funcsettings_table.resizeColumnsToContents()
            
            # connect the funcsettings_table to the update_yaml function
            self.funcsettings_table.itemChanged.connect(lambda item: self.safe_execute(lambda: self.update_yaml(item, func=func)))
            
        
    def profile_macros(self):
        self.dialog = ProfileMacrosDialog()
        self.dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        
        ############ Load profile ############
        # Find the Label current_profile_label
        self.current_profile_label = self.dialog.findChild(QLabel, 'current_profile_label')
        # Set the text of the QLabel to the current profile file in EvaluixConfig
        self.current_profile_label.setText('Current profile: ' + EvaluixConfig['ProfileConfig'])
        # Find the Button btn_load_profile
        self.btn_load_profile = self.dialog.findChild(QPushButton, 'btn_load_profile')
        # Connect the Button to the load_profile function
        self.btn_load_profile.clicked.connect(self.load_profile) #TODO: implement load_profile function
        
        ############ General settings ############
        # Display all information of ProfileConfig in a QTableWidget if the key is not "function_info" or "Macro"
        # Find the QTableWidget in the General settings tab
        self.general_settings_table = self.dialog.findChild(QTableWidget, 'general_settings_table')
        
        # Extract the general settings from ProfileConfig
        general_settings = {key: value for key, value in ProfileConfig.items() if key not in ['function_info', 'Macro']}
        # Set the number of rows in the table to the number of general settings
        self.general_settings_table.setRowCount(len(general_settings))
        # Set the number of columns in the table to 2 (for key and value)
        self.general_settings_table.setColumnCount(2)
        # Set the headers of the table
        self.general_settings_table.setHorizontalHeaderLabels(['Key', 'Value'])
        # Populate the table with the general settings
        for i, (key, value) in enumerate(general_settings.items()):
            # Create QTableWidgetItem objects for the key and value
            key_item = QTableWidgetItem(key)
            value_item = QTableWidgetItem(str(value))
            # Make the key item not editable
            key_item.setFlags(key_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            # Add the items to the table
            self.general_settings_table.setItem(i, 0, key_item)
            self.general_settings_table.setItem(i, 1, value_item)
        
        # Resize the columns to fit their contents
        self.general_settings_table.resizeColumnsToContents()
        
        # connect the general_settings_table to the update_yaml function
        self.general_settings_table.itemChanged.connect(lambda item: self.safe_execute(lambda: self.update_yaml(item, yaml_file=EvaluixConfig['ProfileConfig'])))
        
        ############ Function info ############
        # This is basically a copy of the tabs of the InfoSettingsDialog but instead of a single function, the function can be chosen from a dropdown menu
        # Find the QComboBox in the Function info tab
        self.function_info_combobox = self.dialog.findChild(QComboBox, 'function_combobox')
        # Populate the combobox with the function names in ProfileConfig['function_info']
        self.function_info_combobox.addItems(ProfileConfig['function_info'].keys())
        # connect a selection change to the info_settings function
        self.function_info_combobox.currentTextChanged.connect(self.safe_execute(lambda: self.info_settings(ProfileConfig['function_info'][self.function_info_combobox.currentText()])))
        # Select the function "tan_hyseval" by default, or the first function if it is not available
        if 'tan_hyseval' in ProfileConfig['function_info']:
            self.function_info_combobox.setCurrentText('tan_hyseval')
        else:
            self.function_info_combobox.setCurrentText(list(ProfileConfig['function_info'].keys())[0])
            
        ############ Macros ############
        # TODO: Implement the Macros tab
        
    def load_profile(self):
        # Load the profile from the file dialog
        filename, _ = QFileDialog.getOpenFileName(self, "Load Profile", "", "YAML Files (*.yaml)")
        if filename:
            # Update EvaluixConfig with the new profile file
            EvaluixConfig['ProfileConfig'] = filename
            # Save the updated EvaluixConfig to the config file
            with open(own_path / 'EvaluixConfig.yaml', 'w') as file:
                yaml.dump(EvaluixConfig, file)
                
            # Load the new profile file
            ProfileConfig = load_config(filename)
            
            print(f"Loaded profile from {filename}.")
            
    def convert_value(self, value, expected_type):
        try:
            return expected_type(value)
        except ValueError:
            return None

    def update_yaml(self, item, func=None, key=None, yaml_file=EvaluixConfig['ProfileConfig']):
        if func:
            # Get the row and column of the changed item
            row = item.row()
            column = item.column()

            # Only update the YAML file if the 'Value' column was changed
            if column == 2:
                # Get the parameter name and new value
                name_item = self.settings_table.item(row, 0)
                value_item = self.settings_table.item(row, 2)
                name = name_item.text()
                value = value_item.text()

                if name_item.text() in ['xdata', 'ydata', 'Category']:
                    self.update_status_bar(f"The parameter {name_item.text()} is not allowed to be changed in the YAML file.")

                    return



                # check if input is valid compared to the type of the parameter
                type_mapping = {
                    'int': int,
                    'float': float,
                    'str': str,
                    'list': list,
                    'dict': dict,
                    'Union': Union,
                    'pd.DataFrame': pd.DataFrame,
                    'pd.Series': pd.Series,
                    'np.ndarray': np.ndarray,
                    'bool': str_to_bool,
                    # Add more types here if necessary
                }

                expected_type_str = ProfileConfig['function_info'][func.__name__][name]['type']
                converted_value = None
                if 'Union' in expected_type_str:
                    expected_types = [type_mapping[type_str.strip()] for type_str in expected_type_str.replace('Union[', '').replace(']', '').split(',')]
                    for _expected_type in expected_types:
                        _value = self.convert_value(value, _expected_type)
                        if _value is not None:
                            converted_value = _value
                            break

                else:
                    expected_type = type_mapping.get(expected_type_str)
                    if expected_type is None:
                        self.update_status_bar(f"Unknown type {expected_type_str} for parameter {name}.")
                        return
                    _value = self.convert_value(value, expected_type)
                    if _value is None:
                        self.update_status_bar(f"The input {value} is not valid for the type {expected_type_str}.")

                    converted_value = _value

                # update the value in the table without emitting the signal (otherwise it would call this function again and be trapped in a loop)
                self.settings_table.blockSignals(True) # Block signals
                if converted_value is None:
                    self.settings_table.item(row, 2).setText(ProfileConfig['function_info'][func.__name__][name]['value']) # Reset the item
                    self.settings_table.blockSignals(False) # Unblock signals
                    return
                else:
                    self.settings_table.item(row, 2).setText(str(converted_value)) # Update the item
                    self.settings_table.blockSignals(False) # Unblock signals

                # Update the value in the ProfileConfig dictionary
                ProfileConfig['function_info'][func.__name__][name]['value'] = str(value)
                print(f"Updated {name} to {value} in the ProfileConfig dictionary.")

                # Write the updated ProfileConfig dictionary back to the YAML file
                with open(yaml_file, 'w') as file:
                    # Check if file is available
                    if file:
                        print(f"File {yaml_file} found.")
                    yaml.dump(ProfileConfig, file)

        elif key:
            # in this case the item is directly the entry of the ProfileConfig dictionary
            # for example to change the Autosave_status
            ProfileConfig[key] = item

            with open(yaml_file, 'w') as file:
                yaml.dump(ProfileConfig, file)
        else:
            self.update_status_bar("To update the yaml file, either a key or a func has to be given.")
            return

    def toggle_autosave_status(self):
        ProfileConfig['Autosave_status'] = not ProfileConfig['Autosave_status']

        if ProfileConfig['Autosave_status']:
            if not self.save_path:
                self.save_path = self.open_file_dialog(intention='save', filter='HDF5 files (*.h5)')
                if not self.save_path:
                    self.update_status_bar(saved_message = "Autosave is set to True but no save path is selected. Please select a save path.")
                    return
                else:
                    self.save_progress()
            else:
                self.save_progress()

        self.update_yaml(ProfileConfig['Autosave_status'], key='Autosave_status')
        self.update_status_bar(message=None)

    def save_progress(self, save_path=None, ids=None):
        # Save the progress of the selected datasets to an HDF5 file
        # First check if a valid save path is set. TODO: Implement a "Save As" dialog to change the save path
        
        # This first "if not" checks whether a save path is given as an argument. If not, it checks whether the autosave is enabled and a save path is set.
        if not save_path:
            save_path = self.get_save_path()
            
        # This second "if not" checks whether a save path is now set. If not, it returns and does not save the progress.
        if not save_path:
            self.update_status_bar(saved_message="No progress saved. Please select a valid file.")
            return
        
        self.save_path = save_path # Set the save path (for autosave) to the selected path

        with h5py.File(save_path, 'a') as file:
            self.remove_deleted_datasets(file) # Remove datasets that are not in the global data object
            ids = ids or data.dataset.keys() # Save all datasets if no ids are given

            for i, dataset in enumerate(data): # Iterate over each dataset in the global data object
                if i in list(ids): # Save only the selected datasets (set to all above if no ids are given)
                    self.save_dataset(file, i, dataset) # Save the dataset to the HDF5 file

        self.update_status_bar(saved_message=f"Progress saved to {save_path} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
        self.update_autosave_path(save_path)

    def get_save_path(self):
        # Check if autosave is enabled and a save path is set
        if ProfileConfig['Autosave_status'] and self.save_path:
            return self.save_path
        # Otherwise, open a file dialog to select a save path
        return self.open_file_dialog(intention='save', filter='HDF5 files (*.h5)')

    def remove_deleted_datasets(self, file):
        # Quickly compare if there are any datasets in the file which are not in the global data object and delete them
        # TODO: Think about warning the user that some datasets are not in the global data object and will be deleted before proceeding
        for key in file.keys():
            if int(key.split('_')[-1]) not in data.dataset.keys():
                del file[key]

    def save_dataset(self, file, i, dataset):
        try:
            # Check if the group already exists and delete it
            if f'dataset_{i}' in file:
                del file[f'dataset_{i}']

            # Create a new group for each dataset
            group = file.create_group(f'dataset_{i}')
            # Every attribute of the dataset is saved as an attribute in the group
            for key, value in dataset.items():
                subgroup = self.get_or_create_subgroup(group, key) # Create subgroups for metadata, raw_data and mod_data_n. In the later subgroup, loglist_n and results_n are also saved within mod_data_n.
                self.save_value(subgroup, key, value)
        except Exception as e:
            self.update_status_bar(f"An error occurred while saving the dataset {i}: {e}")

    def get_or_create_subgroup(self, group, key):
        # Check if the current attribute/key is part of the mod_data_n, loglist_n, or results_n data packages
        # These are saved together in the mod_data_n group
        if key.startswith(('mod_data', 'loglist', 'results')):
            n = key.split('_')[-1]
            # create a new mod_data_n group if it doesn't exist
            if f'data_pkg_{n}' not in group:
                return group.create_group(f'data_pkg_{n}')
            # else, choose the existing mod_data_n group
            return group[f'data_pkg_{n}']
        # for metadata, raw_data, and possibly other attributes, create a new subgroup
        return group.create_group(key)

    def save_value(self, subgroup, key, value):
        # This function basically maps the correct hdf5 saving method to the type of the value
        if isinstance(value, pd.DataFrame):
            self.save_dataframe(subgroup, key, value) # For raw_data and mod_data_n
        elif isinstance(value, dict):
            self.save_dict(subgroup, key, value) # For metadata and results_n
        elif isinstance(value, list):
            self.save_list(subgroup, key, value) # For loglist_n
        else:
            subgroup.attrs[key] = value # For all other attributes

    def save_dataframe(self, subgroup, key, value):
        # Save pandas DataFrames as HDF5 datasets in the group. This currently only works for pd.DataFrame objects in which numerically typed columns are stored. TODO: Implement a way to save other types of DataFrames (Kerr images).
        h5dataset = subgroup.create_dataset(key, data=value.to_numpy())
        h5dataset.attrs['columns'] = value.columns.tolist()

    def save_dict(self, subgroup, key, value):
        # There is no default implementation for saving dictionaries in HDF5 files.
        # Therefore, we have to decide how to save the dictionary based on its content/intention.
        if key == 'metadata':
            self.save_metadata(subgroup, value)
        elif 'results' in key: # results_n
            self.save_results(subgroup, key, value)

    def save_metadata(self, subgroup, value):
        # Save dictionaries as groups with attributes in the group
        for k, v in value.items():
            try:
                # If the value is a path and contains backslashes, replace them with forward slashes as backslashes are not allowed in HDF5 attributes
                if isinstance(v, pathlib.Path):
                    v = str(v).replace('\\', '/')
                subgroup.attrs[k] = v
            except TypeError:
                print(f"Could not save {k} with value {v} as attribute in the group metadata.")

    def save_results(self, subgroup, key, value):
        # Save results_n dictionaries as groups with attributes in the group
        # create a subgroup for the main key as every key (fit, analysis method) in the results_n dictionary is a subgroup
        results_subgroup = subgroup.create_group(key)
        for k, v in value.items():
            results_subsubgroup = results_subgroup.create_group(k)
            # Save the results as attributes in the subgroup
            if isinstance(v, dict):
                for kk, vv in v.items():
                    results_subsubgroup.attrs[kk] = vv
            else:
                print(f"Value {v} is not a dictionary. Could not save it as dataset in the group {key}.")

    def save_list(self, subgroup, key, value):
        # Save lists as HDF5 datasets in the group
        subgroup.create_dataset(key, data=value)

    def update_autosave_path(self, save_path):
        if ProfileConfig['Autosave_status'] and self.save_path != save_path:
            reply = QMessageBox.question(self, 'Save Progress',
                                f'Autosave path is currently set to {self.save_path}.\n' \
                                f"Would you like to change the autosave path to {save_path}?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.save_path = str(save_path)

    def load_progress(self, preview=False):
        # Load a saved progress
        global data

        filename = self.open_file_dialog(intention='load', filter='HDF5 files (*.h5)')

        if filename and filename.endswith('.h5'):
            # Open the HDF5 file

            if preview:
                self.hdf5PreviewDialog = HDF5PreviewDialog(filename)
                result = self.hdf5PreviewDialog.exec()

                if result == QDialog.DialogCode.Rejected:
                    self.update_status_bar(f"Cancel button was pressed")
                    return
                
            if data.dataset:
                reply = QMessageBox.question(self, 'Load Progress',
                        "The current data object has datasets inside. " \
                        "Loading progress will delete these datasets. " \
                        "Do you want to continue?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    return
            
            data.clear()
            
            try:
                with h5py.File(filename, 'r') as file:
                    for group_name in file:
                        if group_name.startswith('dataset_'):
                            group = file[group_name]
                            id = int(group_name.split('_')[-1])
                            
                            # Check if the 'raw_data' subgroup and 'columns' attribute exist
                            if 'raw_data' in group and 'columns' in group['raw_data']['raw_data'].attrs:
                                raw_data = pd.DataFrame(np.array(group['raw_data']['raw_data']), columns=list(group['raw_data']['raw_data'].attrs['columns']))
                            else:
                                self.update_status_bar(f"'raw_data' subgroup or 'columns' attribute does not exist in group {group_name}")
                                print(f"'raw_data' subgroup or 'columns' attribute does not exist in group {group_name}")
                                continue
                            
                            data.write_specific_dataset(
                                id,
                                Dataset(
                                    metadata = dict(group['metadata'].attrs),
                                    raw_data = raw_data,
                                )
                            )

                            for subgroup_name in group:
                                if subgroup_name.startswith('data_pkg_'):
                                    key = subgroup_name.split('_')[-1]
                                    subgroup = group[subgroup_name]
                                    
                                    results = {}
                                    for results_key in subgroup[f'results_{key}'].keys():
                                        results[results_key] = dict(subgroup[f'results_{key}'][results_key].attrs)
                                            
                                    data.dataset[id].add_log_mod_results(
                                        key = key,
                                        loglist = list(np.array(subgroup[f'loglist_{key}'])),
                                        mod_data = pd.DataFrame(np.array(subgroup[f'mod_data_{key}']), columns=list(subgroup[f'mod_data_{key}'].attrs['columns'])),
                                        #results = dict(subgroup[f'results_{key}'].attrs),
                                        results = results,
                                    )
                                    
            except Exception as e:
                self.update_status_bar(f"An error occurred while loading the progress: {e}")
                print(f"An error occurred while loading the progress: {e}")
                print(traceback.format_exc())

    def open_FunctionViewer(self):
        # Open the FunctionViewer
        self.functionViewer = FunctionViewer()
        self.functionViewer.applied.connect(self.apply_function)
        self.functionViewer.show()
        
    def apply_function(self, module_name, function_name, parameters):
        # check if module is imported
        if module_name not in sys.modules:
            importlib.import_module(module_name)
            
        # get the function from the module
        func = getattr(sys.modules[module_name], function_name)
        
        # apply the function
        try:
            print(f"Applying function {function_name} with parameters {parameters}")
            for k, v in parameters.items():
                match = re.match(r"data\.dataset\[(\d+)\]\['(.*?)'\]\['(.*?)'\]", v)
                if match:
                    _id, _pkg, _col = match.groups()
                    parameters[k] = data.dataset[int(id)][pkg][col]
                else:
                    parameters[k] = v
            
            results = func(**parameters)
            print(results)
        except Exception as e:
            print(f"An error occurred while applying the function: {e}")
            print(traceback.format_exc())
    
    # 7. Quit the application
    def closeEvent(self, event):
        # Add any cleanup code here if necessary
        # ...

        # Quit the application
        QCoreApplication.quit()
    
    # 8. Insane Clown Posse
    def ICP(self):
        QApplication.instance().setOverrideCursor(Qt.CursorShape.WaitCursor)
        QDesktopServices.openUrl(QUrl("https://www.youtube.com/watch?v=8GyVx28R9-s&t=112"))
        QApplication.instance().restoreOverrideCursor()
    
if __name__ == '__main__':
    try:
        app = QApplication([])
        window = MainWindow()
        window.show()
        app.exec()

    except Exception as e:
        with open(own_path / 'log.txt', 'a') as f:
            f.write(str(e))
            f.write(traceback.format_exc())
        raise e