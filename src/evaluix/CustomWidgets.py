from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QStackedWidget,
    QTabBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QToolButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)
from PyQt6 import uic
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint, QDir
from PyQt6.QtGui import (
    QPainter, 
    QPixmap, 
    QDragEnterEvent, 
    QDropEvent, 
    QDragMoveEvent, 
    QGuiApplication, 
    QFontMetrics, 
    QKeySequence, 
    QIcon, 
    QAction, 
)
from matplotlib.figure import Figure
from matplotlib import cycler
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import Cursor, RangeSlider, RectangleSelector, EllipseSelector, LassoSelector, PolygonSelector
# from qtconsole.rich_jupyter_widget import RichJupyterWidget
# from qtconsole.inprocess import QtInProcessKernelManager
import pathlib
import h5py
import csv
import inspect
import traceback
import importlib
import re
import contextlib
from io import StringIO
import yaml
import numpy as np

# these are the imports for the console widget based on IPython
import subprocess
import IPython
#from IPython.lib.inputhook import inputhook_manager
import threading

#paths
own_path = pathlib.Path(__file__).parent.absolute()
infosettings_path = own_path / "GUIs/Btn_InfoSettings.ui"
profilemacros_path = own_path / "GUIs/ProfileAndMacros.ui"
hdf5preview_path = own_path / "GUIs/HDF5Preview.ui"
manualdatadialog_path = own_path / "GUIs/ManualDataDialog.ui"
functionviewer_path = own_path / "GUIs/FunctionViewer.ui"

QDir.addSearchPath('icons', str(own_path / 'Icons'))

with open(own_path / 'EvaluixConfig.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
with open(own_path / 'Macros.yaml', 'r') as file:
    macros = yaml.safe_load(file)

class ClickableMenu(QMenu):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

class InfoSettingsButton(QPushButton):
    # Define two different signals for the two different button actions
    infoClicked = pyqtSignal()
    buttonClicked = pyqtSignal()
    
    # Initialize and separate the info region from the button
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.info_rect = QRect(self.width() - 16, 0, 16, 16)  # Define the info region
        self.info_rect = QRect(0, 0, 16, 16)  # Define the info region
                
    # Handle the paint event (which happens when the button is drawn)
    def paintEvent(self, event):
        super().paintEvent(event)  # Let the base class handle the event
        self.paintIcon()
        
    # Draw the info icon in the info region
    def paintIcon(self):
        painter = QPainter(self)
        pixmap = QPixmap(str(own_path / "DrawingsAndIcons/icon_infosettings_cropped.png"))  # Load the image
        painter.drawPixmap(self.info_rect, pixmap)  # Draw the image in the info region

    # Handle the mouse press event region dependently
    def mousePressEvent(self, event):
        if self.info_rect.contains(event.pos()):
            self.infoClicked.emit() # Open the info/settings window here
        else:
            self.buttonClicked.emit() # Do the normal button stuff here (data manipulation/evaluation)

    # Handle the resize event, i.e. move the info region to the top right corner if the Window is resized
    def resizeEvent(self, event):
        super().resizeEvent(event)  # Let the base class handle the event
        self.info_rect.moveTopLeft(self.rect().topLeft())  # Move the info region to the top right corner

class InfoSettingsDialog(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi(infosettings_path, self)
        self.setWindowTitle("Info and Settings")
        
class ProfileMacrosDialog(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi(profilemacros_path, self)
        self.setWindowTitle("Profile and Macros")
        
class EditableTabBar(QTabBar):
    def mouseDoubleClickEvent(self, event):
        index = self.tabAt(event.position().toPoint())
        if index >= 0:
            dialog = RenameTabDialog(self.tabText(index), self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.setTabText(index, dialog.get_new_name())

class RenameTabDialog(QDialog):
    def __init__(self, old_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rename Tab")
        self.layout = QVBoxLayout(self)
        self.lineEdit = QLineEdit(old_name, self)
        self.layout.addWidget(self.lineEdit)
        self.okButton = QPushButton("OK", self)
        self.okButton.clicked.connect(self.accept)
        self.layout.addWidget(self.okButton)
        self.cancelButton = QPushButton("Cancel", self)
        self.cancelButton.clicked.connect(self.reject)
        self.layout.addWidget(self.cancelButton)

    def get_new_name(self):
        return self.lineEdit.text()

# Custom QTableWidget with a context menu that allows the user to export the table data to a CSV file or copy it to the clipboard
class ExportTableWidget(QTableWidget):
    
    # Initialize the custom table widget
    def __init__(self, parent=None):
        # Call the base class constructor, i.e. take everything from the base class
        super().__init__(parent)
        # Set the context menu policy to custom, so we can show the context menu when the user right-clicks the table
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        
        # Create context menus for the headers
        self.horizontalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.horizontalHeader().customContextMenuRequested.connect(self.showHeaderContextMenu)
        self.verticalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.verticalHeader().customContextMenuRequested.connect(self.showHeaderContextMenu)
        
        # Enable column reordering
        self.horizontalHeader().setSectionsMovable(True)

    # Show the context menu when the user right-clicks the table
    def showContextMenu(self, pos):
        contextMenu = self.createContextMenu()
        # Show the context menu at the specified position (mouse cursor position)
        contextMenu.exec(self.mapToGlobal(pos))

    # Show the same context menu for the horizontal and vertical headers
    def showHeaderContextMenu(self, pos):
        contextMenu = self.createContextMenu()
        # Show the context menu at the specified position (mouse cursor position)
        contextMenu.exec(self.mapToGlobal(pos))

    def createContextMenu(self):
        # Create a context menu with the standard actions (cut, copy, paste, etc.)
        contextMenu = QMenu(self)
        # Add custom actions to the context menu
        exportselectedAction = contextMenu.addAction("Export selected to CSV", lambda: self.export_to_csv('selected'))
        exporttableAction = contextMenu.addAction("Export table to CSV", lambda: self.export_to_csv('all'))
        copyselectedAction = contextMenu.addAction("Copy selected to Clipboard", lambda: self.copy_to_clipboard('selected'))
        copytableAction = contextMenu.addAction("Copy table to Clipboard", lambda: self.copy_to_clipboard('all'))
        # Return the context menu so that it can be shown
        return contextMenu
        
    # Export the table data to a CSV file
    def export_to_csv(self, mode='selected'):
        # Open a file dialog to get the file name and location
        filename, _ = QFileDialog.getSaveFileName(self, "Save File", "", "CSV Files (*.csv)")
        if filename:
            try:
                with open(filename, 'w', newline='') as file:
                    writer = csv.writer(file, delimiter=' ; ')
                    writer.writerows(self.get_data(mode))
            except Exception as e:
                return QMessageBox.critical(self, "Error", f"An error occurred while exporting the data: {e}")
                
    # Copy the selected table data to the clipboard
    def copy_to_clipboard(self, mode='selected'):
        try:
            clipboard = QApplication.clipboard() # Get the clipboard
            text = "\n".join("\t".join(row) for row in self.get_data(mode))
            clipboard.setText(text)
        except Exception as e:
            return QMessageBox.critical(self, "Error", f"An error occurred while copying the data to the clipboard: {e}")
        
    # Get the selected or entire table data
    def get_data(self, mode: str = 'selected') -> list[list[str]]:
        """
        Gets the headers and items in the Table to export them to a CSV file or to the clipboard.
        The headers are the horizontal and vertical headers of the Table, the items are the content of the Table.
        The mode determines whether the whole table is exported or only the selected items.

        Parameters
        ----------
        mode : str
            'selected' or 'all'. If 'selected', only the selected items are exported, if 'all', the whole table is exported.

        Returns
        -------
        data : list
            A list of lists containing the headers and items of the Table. The list is structured as follows:
            - The first list contains the headers of the Table (horizontal headers).
            - The following lists contain the items of the Table (content of the Table). Each list represents a row of the Table.

        """
        data = [] # Create an empty list to store the table data
        
        if mode == 'selected': # mode == 'selected'
            indexes = self.selectedIndexes()
            if indexes:
                # Sort the indexes first by row, then by column
                indexes.sort(key=lambda index: (index.row(), index.column()))
                # Get the row data for the first index and the headers
                unique_rows = list(set(index.row() for index in indexes))
                unique_cols = list(set(index.column() for index in indexes))
                
                # sort the indexes
                unique_rows.sort()
                unique_cols.sort()
                
                # Get the headers
                headers = ['Row Nr.'] + [self.horizontalHeaderItem(column).text() if self.horizontalHeaderItem(column) else "" for column in unique_cols]
                data.append(headers)
                
                for row in unique_rows:
                    vertical_header_item = self.verticalHeaderItem(row)
                    row_data = [vertical_header_item.text() if vertical_header_item else str(row + 1)]
                    
                    # Iterate over the unique columns
                    for column in unique_cols:
                        item = self.item(row, column)
                        row_data.append(item.text() if item else "")
                    data.append(row_data)
        else:  # mode == 'all'
            # Get the headers and add them to the data
            headers = ['Row Nr.'] + [self.horizontalHeaderItem(i).text() for i in range(self.columnCount())]
            data.append(headers)
            for i in range(self.rowCount()):
                vertical_header_item = self.verticalHeaderItem(i)
                row_data = [vertical_header_item.text() if vertical_header_item else str(i+1)]
                for j in range(self.columnCount()):
                    try:
                        row_data.append(self.item(i, j).text())
                    except AttributeError:
                        row_data.append("")
                data.append(row_data)
        return data

class MacroEditorDialog(QDialog):
    def __init__(self, macros):
        super().__init__()
        self.macros = macros
        self.setWindowTitle("Macro Editor")
        self.layout = QVBoxLayout(self)
        self.stackedWidget = MacroStackedWidgetContainer(macros=self.macros, table=True)
        self.layout.addWidget(self.stackedWidget)
        
        # Add a button to save the changes
        self.saveButton = QPushButton("Save Changes")
        self.saveButton.clicked.connect(self.saveChanges)
        self.layout.addWidget(self.saveButton)
        
        # Add a button to cancel the changes
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.clicked.connect(self.reject)
        self.layout.addWidget(self.cancelButton)
        
        # Add an "add page" and "delete page" button
        self.addPageButton = QPushButton("Add Page")
        self.addPageButton.clicked.connect(self.stackedWidget.stackedWidget.addNewPage)
        self.layout.addWidget(self.addPageButton)
        self.deletePageButton = QPushButton("Delete Page")
        self.deletePageButton.clicked.connect(self.stackedWidget.stackedWidget.deleteCurrentPage)
        self.layout.addWidget(self.deletePageButton)
        
    def saveChanges(self):
        # Save the changes to the macros dictionary
        self.accept()
        
    def reject(self):
        # Close the dialog without saving the changes
        self.accept()
    
    def accept(self):
        # Close the dialog and save the changes
        super().accept()

class MacroTableWidget(QTableWidget):
    def __init__(self, parent=None, macros=None, page_name=None):
        super(MacroTableWidget, self).__init__(parent)
        self.macros = macros
        self.page_name = page_name
        # 0 rows, 4 columns initially
        self.setRowCount(0)
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["Active", "Function Type", "Function", "Parameters"])

        # Set the table to fill the widget
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # config has the structure: {func.__name__: {'Category': 'Data Manipulation' or 'Data Evaluation'}}
        # Invert the config dictionary to get the functions by type
        self.functions_by_type = {}
        for func_name, func_info in config.items():
            if isinstance(func_info, dict) and 'Category' in func_info:
                func_type = func_info['Category']
                if func_type not in self.functions_by_type:
                    self.functions_by_type[func_type] = []
                self.functions_by_type[func_type].append(func_name)
        
        # Now a similar game as in the previous example but for the function parameters. These are all stored in the config file
        # and are represented by all keys which are not 'Category' or 'ydata'. Here, I need a dictionary with the function name as key
        # and a string containing all parameters in the structure 'parameter1, default1; parameter2, default2; ...'
        self.function_parameters = {}
        for func_name, func_info in config.items():
            if isinstance(func_info, dict) and 'Category' in func_info:
                parameters = [f"{param}, {info['default']}" for param, info in func_info.items() if param not in ['Category', 'ydata']]
                self.function_parameters[func_name] = "; ".join(parameters)
        
        for i in range(4):
            self.addRow()  # Add initial rows

        # Connect the cellChanged signal to the resizeColumns slot
        self.cellChanged.connect(self.resizeColumns)
        self.cellChanged.connect(self.updateMacrosDictionary)
        
        # Apply custom stylesheet for the header
        self.setStyleSheet("""
            QHeaderView::section {
                background-color: rgb(45, 55, 65);
            }
        """)

    def addRow(self):
        rowPosition = self.rowCount()
        self.insertRow(rowPosition)

        # Add a checkbox to the first column
        checkbox = QCheckBox()
        self.setCellWidget(rowPosition, 0, checkbox)

        # Add a combobox to the second column
        typeComboBox = QComboBox()
        typeComboBox.addItems(self.functions_by_type.keys())
        typeComboBox.currentIndexChanged.connect(lambda: self.updateFunctionComboBox(rowPosition))
        self.setCellWidget(rowPosition, 1, typeComboBox)

        # Add a combobox to the third column
        functionComboBox = QComboBox()
        functionComboBox.currentIndexChanged.connect(lambda: self.updateParameterLabel(rowPosition))
        self.setCellWidget(rowPosition, 2, functionComboBox)
        
        # Add an empty item to the fourth column
        self.setItem(rowPosition, 3, QTableWidgetItem(""))

        # Resize columns to fit content after adding a row
        self.resizeColumns()

    def updateFunctionComboBox(self, row):
        typeComboBox = self.cellWidget(row, 1)
        functionComboBox = self.cellWidget(row, 2)
        selected_type = typeComboBox.currentText()
        
        # Clear the function combobox
        functionComboBox.clear()
        
        # Populate the function combobox with functions of the selected type
        if selected_type in self.functions_by_type:
            functionComboBox.addItems(self.functions_by_type[selected_type])
            
    def updateParameterLabel(self, row):
        functionComboBox = self.cellWidget(row, 2)
        selected_function = functionComboBox.currentText()
        
        if selected_function in self.function_parameters:
            # Get the function parameters from the dictionary
            parameters = self.function_parameters[selected_function]
            # Set the parameters as the item text in the fourth column
            self.setItem(row, 3, QTableWidgetItem(parameters))
    
    def resizeColumns(self):
        self.resizeColumnsToContents()

    def updateMacrosDictionary(self):
        if self.page_name not in self.macros:
            self.macros[self.page_name] = {}
        for row in range(self.rowCount()):
            self.macros[self.page_name][row] = {
                "Active": self.cellWidget(row, 0).isChecked(),
                "Function Type": self.cellWidget(row, 1).currentText(),
                "Function": self.cellWidget(row, 2).currentText(),
                "Parameters": self.item(row, 3).text() if self.item(row, 3) else ""
            }

class MacroStackedWidget(QStackedWidget):
    def __init__(self, parent=None, macros=None, table=False):
        super(MacroStackedWidget, self).__init__(parent)
        self.pageNames = {}  # Dictionary to store page names

        if macros is not None:
            self.macros = macros['Hysteresis']
        else:
            self.macros = {}

        # Add initial pages
        try:
            for key in self.macros.keys():
                if not table:
                    self.addNewPage(name=key)
                elif table:
                    self.addNewPage(name=key, table=True)
        except KeyError as e:
            print(f"KeyError: {e}")

        # Connect the currentChanged signal to update the page label
        self.currentChanged.connect(self.updatePageComboBox)

    def addNewPage(self, name=None, table=False):
        # Create a new page
        newPage = QWidget()
        layout = QVBoxLayout(newPage)
        
        if not table:
            # Add a quick overview about the macro to the new page
            # So: number of lines/functions and then further clarified as number of unknown/Data Manipulation/Data Evaluation functions
            try:
                macro_dict = self.macros[name]
                nr_functions = len([macro for macro in macro_dict.values() if macro['Active']])
                nr_unknown = len([macro for macro in macro_dict.values() if macro['Category'] == 'unknown' and macro['Active']])
                nr_data_manipulation = len([macro for macro in macro_dict.values() if macro['Category'] == 'Data Manipulation' and macro['Active']])
                nr_data_evaluation = len([macro for macro in macro_dict.values() if macro['Category'] == 'Data Evaluation' and macro['Active']])
            except KeyError as e:
                print(f"KeyError: {e}")
                nr_functions = nr_unknown = nr_data_manipulation = nr_data_evaluation = 0
            
            label = QLabel(f"Number of functions: {nr_functions}\n- Unknown: {nr_unknown}\n- Manipulation: {nr_data_manipulation}\n- Evaluation: {nr_data_evaluation}")
            layout.addWidget(label)
            layout.setContentsMargins(0, 0, 0, 0)  # Set padding to zero for the new page layout
            
        elif table:
            # Add a table widget to the new page
            tableWidget = MacroTableWidget(macros=self.macros, page_name=name)
            layout.addWidget(tableWidget)
            layout.setContentsMargins(0, 0, 0, 0)  # Set padding to zero for the new page layout

        # Add the new page to the QStackedWidget
        self.addWidget(newPage)
        self.pageNames[self.count() - 1] = name  # Store the page name in the dictionary
        self.setCurrentWidget(newPage)
        self.updatePageComboBox()

    def deleteCurrentPage(self):
        # Remove the current page if it's not one of the first two
        currentIndex = self.currentIndex()
        if currentIndex > 1:
            self.removeWidget(self.widget(currentIndex))
            del self.pageNames[currentIndex]  # Remove the page name from the dictionary

        # If there are no pages left, create a new one
        if self.count() == 1:
            self.addNewPage()

    def updatePageComboBox(self):
        # Update the combo box to show the current page names
        comboBox = self.parent().pageComboBox
        comboBox.blockSignals(True)  # Block signals to prevent recursive updates
        comboBox.clear()
        for i in range(self.count()):
            comboBox.addItem(self.pageNames.get(i, f"Page nr {i + 1}"))
        comboBox.setCurrentIndex(self.currentIndex())
        comboBox.blockSignals(False)  # Unblock signals

        # Update the current position label
        currentIndex = self.currentIndex() + 1
        totalPages = self.count()
        self.parent().positionLabel.setText(f"{currentIndex}/{totalPages}")

class MacroStackedWidgetContainer(QWidget):
    def __init__(self, parent=None, macros=macros, table=False):
        super(MacroStackedWidgetContainer, self).__init__(parent)

        # Initialize the macros dictionary
        self.macros = macros

        # Combo box to select the current page
        self.pageComboBox = QComboBox()

        # Add Macros Label
        self.macrosLabel = QLabel("Macros")
        self.macrosLabel.adjustSize()
        self.macrosLabel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Previous page button
        self.prevButton = QPushButton("<")
        self.prevButton.setFixedSize(20, 20)
        self.prevButton.clicked.connect(self.prevPage)

        # Next page button
        self.nextButton = QPushButton(">")
        self.nextButton.setFixedSize(20, 20)
        self.nextButton.clicked.connect(self.nextPage)

        # Label to display the current position
        self.positionLabel = QLabel("1/1")
        self.positionLabel.adjustSize()
        self.positionLabel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Create a QWidget to hold the combo box and buttons
        self.cornerWidget = QWidget()
        self.cornerLayout = QHBoxLayout(self.cornerWidget)
        self.cornerLayout.addWidget(self.macrosLabel)
        self.cornerLayout.addWidget(self.prevButton)
        self.cornerLayout.addWidget(self.positionLabel)
        self.cornerLayout.addWidget(self.pageComboBox)
        self.cornerLayout.addWidget(self.nextButton)
        spacer = QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.cornerLayout.addItem(spacer)

        # Set layout margins to zero
        self.cornerLayout.setContentsMargins(0, 0, 0, 0)

        # Create a main layout to include the cornerWidget and QStackedWidget
        self.mainLayout = QGridLayout(self)
        self.mainLayout.addWidget(self.cornerWidget, 0, 0, 1, 4)

        if not table: # Overview mode in the main window
            # Create the QStackedWidget
            self.stackedWidget = MacroStackedWidget(self, macros=self.macros)
            self.mainLayout.addWidget(self.stackedWidget, 1, 0, 1, 3)
            
            # Create an "edit" button to edit the macros
            self.editButton = QPushButton("Edit\nMacros")
            self.editButton.clicked.connect(self.editMacros)
            self.mainLayout.addWidget(self.editButton, 1, 3, 1, 1)

            # Connect the combo box to the QStackedWidget, which is only possible after both are created
            self.pageComboBox.currentIndexChanged.connect(self.changePage)

        else: # Detailed table mode in the macro editor dialog
            # Create the QStackedWidget
            self.stackedWidget = MacroStackedWidget(self, macros=self.macros, table=True)
            self.mainLayout.addWidget(self.stackedWidget, 1, 0, 1, 4)
            
            # Connect the combo box to the QStackedWidget, which is only possible after both are created

        # Set the main layout
        self.setLayout(self.mainLayout)
        
    def changePage(self, index):
        print(f"Changing to page {index + 1}")
        self.stackedWidget.setCurrentIndex(index)

    def prevPage(self):
        currentIndex = self.stackedWidget.currentIndex()
        if currentIndex > 0:
            self.stackedWidget.setCurrentIndex(currentIndex - 1)
        elif currentIndex == 0:
            self.stackedWidget.setCurrentIndex(self.stackedWidget.count() - 1)

    def nextPage(self):
        currentIndex = self.stackedWidget.currentIndex()
        if currentIndex < self.stackedWidget.count() - 1:
            self.stackedWidget.setCurrentIndex(currentIndex + 1)
        elif currentIndex == self.stackedWidget.count() - 1:
            self.stackedWidget.setCurrentIndex(0)

    def editMacros(self):
        # Open the macro editor dialog
        dialog = MacroEditorDialog(self.macros)
        dialog.exec()        

# Custom QTableWidgetItem that sorts numeric values correctly
class NumericTableWidgetItem(QTableWidgetItem):
    def __lt__(self, other):
        return float(self.text()) < float(other.text())

class DragDropTableWidget(ExportTableWidget):
    # Define a custom signal that will be emitted when a file is dropped, so this class can be connected to the main window
    fileDropped = pyqtSignal(str)
    
    # Define a custom table widget that can accept drag and drop events
    def __init__(self, parent=None, combobox = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        # the combobox is used to check for "Kerr_imgs" in which dropping folders will emit the folders path instead of all files in the folder
        self.combobox = combobox

    # Handle the drag and drop events
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls(): # check if the dragged object is a file (by the url)
            event.acceptProposedAction()

    # Handle the drag move event
    def dragMoveEvent(self, event: QDragMoveEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    # def dropEvent(self, event: QDropEvent):
    #     if event.mimeData().hasUrls():
    #         # Iterate over all URLs
    #         for url in event.mimeData().urls():
    #             file_path = pathlib.Path(url.toLocalFile())
    #             # Check if the file path is a directory
    #             if file_path.is_dir():
    #                 # Check for the specific case
    #                 if self.comboBox is not None and self.comboBox.currentText() == "Kerr_imgs":
    #                     # If it's the specific case, emit the directory path
    #                     self.fileDropped.emit(str(file_path))
    #                 else:
    #                     # If it's not the specific case, iterate over all files in the directory
    #                     for sub_path in file_path.iterdir():
    #                         # Only process files, not subdirectories
    #                         if sub_path.is_file():
    #                             # Emit the custom signal with the file path as an argument
    #                             self.fileDropped.emit(str(sub_path))
    #             else:
    #                 # If it's not a directory, emit the custom signal with the file path as an argument
    #                 self.fileDropped.emit(str(file_path))
    
    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            # Iterate over all URLs
            for url in event.mimeData().urls():
                file_path = pathlib.Path(url.toLocalFile())
                # Emit the custom signal with the file or directory path as an argument
                self.fileDropped.emit(str(file_path))
        
class CustomNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas, parent, *args, **kwargs):
        super(CustomNavigationToolbar, self).__init__(canvas, parent, *args, **kwargs)
        
        # Add a dropdown menu to select the selector
        self.selector_menu = QMenu(self)
        self.add_selector_action('Rectangle Selector', 'rectangle')
        self.add_selector_action('Ellipse Selector', 'ellipse')
        self.add_selector_action('Polygon Selector', 'polygon')
        self.add_selector_action('Lasso Selector', 'lasso')

        # Create a tool button and set its menu
        self.selector_button = QToolButton(self)
        self.selector_button.setMenu(self.selector_menu)
        self.selector_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.selector_button.setText('Selector*')

        # Add the tool button to the toolbar
        self.addWidget(self.selector_button)
        
        # Add a button to toggle the cursor
        self.cursor_action = QAction('Toggle Cursor', self)
        self.updateCursorActionStyle()
        self.cursor_action.setCheckable(True)
        self.cursor_action.setChecked(True)
        # get rid of the button changing color when checked

        self.cursor_action.toggled.connect(self.toggleCursor)
        self.addAction(self.cursor_action)
    
    def updateCursorActionStyle(self):
        # Find the QToolButton for the QAction
        tool_button = self.widgetForAction(self.cursor_action)
        if tool_button:
            tool_button.setStyleSheet("""
                QToolButton {
                    background-color: rgb(45, 55, 65);
                    color: white;
                    border: 1px solid black;
                }
                QToolButton:hover {
                    background-color: rgb(65, 75, 85);
                }
                QToolButton:checked {
                    background-color: rgb(25, 35, 45);
                }
            """)

    def add_selector_action(self, text, selector_type):
        action = QAction(text, self)
        action.triggered.connect(lambda checked, st=selector_type: self.setSelector(st))
        self.selector_menu.addAction(action)

    def setSelector(self, selector_type):
        # Get the mouse button state
        mouse_buttons = QApplication.mouseButtons()

        if mouse_buttons == Qt.MouseButton.RightButton:
            # Right-click: delete all old selectors and create a new one
            self.canvas.selectors.clear()
            self.canvas.setSelector(selector_type)
        elif mouse_buttons == Qt.MouseButton.LeftButton:
            # Left-click: add a new selector
            self.canvas.setSelector(selector_type)

        # Deactivate the cursor
        self.canvas.cursor.set_active(False)
        self.cursor_action.setChecked(False)

    def toggleCursor(self, checked):
        # Set the cursor's active state
        self.canvas.cursor.set_active(checked)

        # Deactivate the selectors if the cursor is active
        if checked:
            for selector in self.canvas.selectors.values():
                selector.set_active(False)

class CustomCursor:
    def __init__(self, axes, useblit=True, color='k', linestyle='-.', linewidth=1):
        self.cursor = Cursor(axes, useblit=useblit, color=color, linestyle=linestyle, linewidth=linewidth)
        self.active = False

    def set_active(self, active):
        self.active = active
        self.cursor.visible = active
        self.cursor.canvas.draw_idle()

class CustomSelector:
    def __init__(self, axes, selector_type, on_select):
        self.axes = axes
        self.selector_type = selector_type
        self.on_select = on_select
        self.selector = self.create_selector(selector_type)

    def create_selector(self, selector_type):
        if selector_type == 'rectangle':
            return RectangleSelector(self.axes, self.on_select, useblit=True, button=[1, 3], interactive=True)
        elif selector_type == 'ellipse':
            return EllipseSelector(self.axes, self.on_select, useblit=True, button=[1, 3], interactive=True)
        elif selector_type == 'polygon':
            return PolygonSelector(self.axes, self.on_select, useblit=True)
        elif selector_type == 'lasso':
            return LassoSelector(self.axes, self.on_select, useblit=True, button=[1, 3])
        else:
            raise ValueError(f"Unknown selector type: {selector_type}")

    def set_active(self, active):
        if self.selector:
            self.selector.set_active(active)

class ResultsTable(QWidget):
    def __init__(self):
        super().__init__()

        # Create a vertical layout to hold the results
        self.vertical_layout = QVBoxLayout(self)
        self.grid_layout = QGridLayout()
        
        self.vertical_layout.addLayout(self.grid_layout)
        self.vertical_layout.addStretch(1)  # Add stretchable space at the bottom
        
        self.setLayout(self.vertical_layout)
        self.show()

    def fill_widget(self, params):
        self.params = params

        # Clear existing rows except headers
        self.clear()

        # Add headers
        self.grid_layout.addWidget(QLabel('Key', self), 0, 0)
        self.grid_layout.addWidget(QLabel('Value', self), 0, 1)
        self.grid_layout.addWidget(QLabel('Uncertainty', self), 0, 2)
        self.grid_layout.addWidget(QLabel('Rel. unc. (%)', self), 0, 3)
        self.grid_layout.addWidget(QLabel('Display', self), 0, 4)
        
        # Add a horizontal line to separate headers and data
        line = QFrame(self)
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self.grid_layout.addWidget(line, 1, 0, 1, 6)

        # Add results
        row = 2
        for key, value in self.params.items():
            if not key.startswith('d'):
                # This approach means uncertainty = 0 or N/A if there is no uncertainty or if it is named in another way
                uncertainty_key = 'd' + key
                uncertainties = self.params.get(uncertainty_key, 'N/A')
                
                key_tooltip = self.get_tooltip(key)
                
                # Ensure values and uncertainties are iterable
                if not isinstance(value, (list, tuple)):
                    value = [value]
                if not isinstance(uncertainties, (list, tuple)):
                    uncertainties = [uncertainties]
                
                # Display each value and its corresponding uncertainty
                key_displayed = False
                for val, unc in zip(value, uncertainties):
                    if isinstance(val, dict):  # Handle dictionary values
                        for val_key, val_val in val.items():
                            formatted_val, formatted_unc, formatted_unc_percentage = self.format_value(val_val, 'N/A')
                            key_label = self.add_interactable_label(f"{key}.{val_key}", row, 0)
                            key_label.setToolTip(key_tooltip)
                            self.add_interactable_label(formatted_val, row, 1)
                            self.add_interactable_label(formatted_unc, row, 2)
                            self.add_interactable_label(formatted_unc_percentage, row, 3)
                            self.grid_layout.addWidget(QLabel('', self), row, 4)
                            row += 1
                    else:
                        formatted_val, formatted_unc, formatted_unc_percentage = self.format_value(val, unc)
                        if not key_displayed:
                            key_label = self.add_interactable_label(key, row, 0)
                            key_label.setToolTip(key_tooltip)
                            try:
                                self.add_interactable_label(formatted_val, row, 1)
                                self.add_interactable_label(formatted_unc, row, 2)
                                self.add_interactable_label(formatted_unc_percentage, row, 3)
                            except (TypeError, ValueError):
                                self.add_interactable_label('unknown', row, 1)
                                self.add_interactable_label('unknown', row, 2)
                                self.add_interactable_label('unknown', row, 3)
                            
                            # Add a checkbox for display option
                            display_checkbox = QCheckBox(self)
                            self.grid_layout.addWidget(display_checkbox, row, 4, 1, 1)
                            
                            key_displayed = True
                        else:
                            self.grid_layout.addWidget(QLabel('', self), row, 0)
                            try:
                                self.add_interactable_label(formatted_val, row, 1)
                                self.add_interactable_label(formatted_unc, row, 2)
                                self.add_interactable_label(formatted_unc_percentage, row, 3)
                            except (TypeError, ValueError):
                                self.add_interactable_label('unknown', row, 1)
                                self.add_interactable_label('unknown', row, 2)
                                self.add_interactable_label('unknown', row, 3)
                            self.grid_layout.addWidget(QLabel('', self), row, 4)
                        
                        row += 1
        self.show()

    def get_tooltip(self, key):
        tooltips = {
            'r_squared': 'The coefficient of determination (R^2) is a statistical measure of how close the data are to the fitted regression line.',
            'HEB': 'The Exchange Bias Field (HEB) is the macroscopic shift of the hysteresis loop along the field axis.',
            'HEB1': 'The Exchange Bias Field (HEB) is the macroscopic shift of the hysteresis loop along the field axis. Here of the left loop.',
            'HEB2': 'The Exchange Bias Field (HEB) is the macroscopic shift of the hysteresis loop along the field axis. Here of the right loop.',
            'HC': 'The Coercivity (HC) is the absolute field difference to the HEB field at which the magnetization is zero. This it is half the loop width.',
            'HC1': 'The Coercivity (HC) is the absolute field difference to the HEB field at which the magnetization is zero. This it is half the loop width. Here of the left loop.',
            'HC2': 'The Coercivity (HC) is the absolute field difference to the HEB field at which the magnetization is zero. This it is half the loop width. Here of the right loop.',
            'MR': 'The remanent magnetization (MR) is the averaged absolute magnetization of the sample at zero field after saturation.',
            'MHEB': 'The magnetization at the HEB field (MHEB) is the averaged absolute magnetization of the sample at the HEB field.',
            'MHEB1': 'The magnetization at the HEB field (MHEB) is the averaged absolute magnetization of the sample at the HEB field. Here of the left loop.',
            'MHEB2': 'The magnetization at the HEB field (MHEB) is the averaged absolute magnetization of the sample at the HEB field. Here of the right loop.',
            'integral': 'The integral is the area under the hysteresis loop(s). The integral is meaningless for non-absolute values.',
            'integral1': 'The integral is the area under the left hysteresis loop.',
            'integral2': 'The integral is the area under the right hysteresis loop.',
            'saturation_fields': 'The saturation fields are the field values at which the magnetization reaches (95% of) its maximum.',
            'saturation_fields1': 'The saturation fields are the field values at which the magnetization reaches (95% of) its maximum. Here of the left loop.',
            'saturation_fields2': 'The saturation fields are the field values at which the magnetization reaches (95% of) its maximum. Here of the right loop.',
            'slope_atHC': 'The slope at the coercivity (HC) is the slope of the magnetization curve at the coercivity field.',
            'slope_atHC1': 'The slope at the coercivity (HC) is the slope of the magnetization curve at the coercivity field. Here of the left loop.',
            'slope_atHC2': 'The slope at the coercivity (HC) is the slope of the magnetization curve at the coercivity field. Here of the right loop.',
            'slope_atHEB': 'The slope at the HEB field is the slope of the magnetization curve at the HEB field.',
            'slope_atHEB1': 'The slope at the HEB field is the slope of the magnetization curve at the HEB field. Here of the left loop.',
            'slope_atHEB2': 'The slope at the HEB field is the slope of the magnetization curve at the HEB field. Here of the right loop.',
            'alpha': 'The angle alpha is the angle between the slopes at the coercive field and the HEB field. For a rectangular loop, alpha is 90°.',
            'alpha1': 'The angle alpha is the angle between the slopes at the coercive field and the HEB field. For a rectangular loop, alpha is 90°. Here of the left loop.',
            'alpha2': 'The angle alpha is the angle between the slopes at the coercive field and the HEB field. For a rectangular loop, alpha is 90°. Here of the right loop.',
            'rectangularity': 'The rectangularity is the sinus of the angle alpha. For a rectangular loop, the rectangularity is 1.',
            'rectangularity1': 'The rectangularity is the sinus of the angle alpha. For a rectangular loop, the rectangularity is 1. Here of the left loop.',
            'rectangularity2': 'The rectangularity is the sinus of the angle alpha. For a rectangular loop, the rectangularity is 1. Here of the right loop.',
            'ratio': 'The ratio of a certain quantity with loop_left/loop_right. For more information, see the tooltip of the specific quantity.',
            'xunit': 'The unit of the x-axis.',
            'yunit': 'The unit of the y-axis.',
        }

        return tooltips.get(key, 'No tooltip available')

    def add_interactable_label(self, text, row, column):
        label = QLabel(text, self)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        label.customContextMenuRequested.connect(lambda pos, lbl=label: self.show_context_menu(pos, lbl))
        self.grid_layout.addWidget(label, row, column)
        return label

    def show_context_menu(self, pos, label):
        menu = QMenu(self)
        copy_action = QAction('Copy', self)
        copy_action.triggered.connect(lambda: self.copy_to_clipboard(label.text()))
        menu.addAction(copy_action)
        
        copy_all_action = QAction('Copy All', self)
        copy_all_action.triggered.connect(lambda: self.copy_to_clipboard(self.params))
        menu.addAction(copy_all_action)
        
        menu.exec(label.mapToGlobal(pos))

    def copy_to_clipboard(self, text_or_dict):
        if isinstance(text_or_dict, dict):
            text = '\n'.join([f"{key}: {value}" for key, value in text_or_dict.items()])
        else:
            text = text_or_dict
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

    def format_value(self, value, uncertainty):
        # Helper function to slice a string of a float to 5 digits
        def slice_to_5_digits(s):
            digits = 0
            result = []
            for char in s:
                if char.isdigit():
                    digits += 1
                result.append(char)
                if digits == 5:
                    break
            return ''.join(result)
        
        # Check if value is not None
        if value is None:
            value = 'unknown'
            uncertainty = 'N/A'
            uncertainty_percentage = 'N/A'
            return value, uncertainty, uncertainty_percentage

        # Check if uncertainty is not None. If it is, set it to "N/A"
        if uncertainty is None:
            uncertainty = 'N/A'
            uncertainty_percentage = 'N/A'
        
        try:
            # Try to convert value and uncertainty to float
            value = float(value)
            uncertainty = float(uncertainty)
            uncertainty_percentage = self.calculate_uncertainty_percentage(value, uncertainty)
            # If this worked, try to get the order of magntiude of both values
            if abs(value) == float('inf') or abs(uncertainty) == float('inf'):
                raise ValueError('Value is infinite')
            else:
                value_order = int(f'{value:e}'.split('e')[1])
                uncertainty_order = int(f'{uncertainty:e}'.split('e')[1])

            # The uncertainty order determines the number of decimal places to show. If the uncertainty is between 1e-3 and 1e3, show 5 digit floats for both,
            # value and uncertainty only exception is when the value exceeds 1e3, then, as well as when the uncertainty is below 1e-3 or above 1e3, show 4 
            # digits for both value and uncertainty in scientific notation
            if -3 <= uncertainty_order <= 3 and not value_order > 3:
                value_str = f'{value:.10f}'
                uncertainty_str = f'{uncertainty:.10f}'
                
                value = slice_to_5_digits(value_str)
                uncertainty = slice_to_5_digits(uncertainty_str)
            else:
                value = f'{value:.4e}'
                uncertainty = f'{uncertainty:.4e}'

        # If conversion fails, set value and uncertainty to the string representation of the value
        # For example for the key xunit, the value is a string and cannot be converted to float
        except ValueError:
            try:
                # Try to get just the value (for example for r_squared where the value is given and the uncertainty is None)
                value = float(value)
                uncertainty = str(uncertainty)
                uncertainty_percentage = 'N/A'
                if abs(value) == float('inf'):
                    raise ValueError('Value is infinite')
                else:
                    value_order = int(f'{value:e}'.split('e')[1])

                if -3 <= value_order <= 3:
                    value_str = f'{value:.10f}'
                    value = slice_to_5_digits(value_str)

            except ValueError:
                try: # Try to convert the value to a string
                    value = str(value)
                    uncertainty = str(uncertainty)
                except ValueError: # If this fails, raise an error
                    raise ValueError(f'Could not convert value "{value}" and uncertainty "{uncertainty}" to float')

        return value, uncertainty, uncertainty_percentage

    def calculate_uncertainty_percentage(self, value, uncertainty):
        def slice_to_5_digits(s):
            digits = 0
            result = []
            for char in s:
                if char.isdigit():
                    digits += 1
                result.append(char)
                if digits == 5:
                    break
            return ''.join(result)
        
        try:
            value = float(value)
            uncertainty = float(uncertainty)
            if value != 0:
                percentage = (uncertainty / abs(value)) * 100
                # if the percentage is between 1e-3 and 1e3, show 5 digit floats, otherwise show 4 digit floats in scientific notation
                if 1e-3 <= percentage <= 1e3:
                    percentage_str = f'{percentage:.10f}'
                    percentage = slice_to_5_digits(percentage_str)
                else:
                    percentage = f'{percentage:.4e}'
                
                return percentage
            else:
                return 'N/A'
        except (ValueError, ZeroDivisionError):
            return 'N/A'

    def clear(self):
        # Remove all widgets except the headers and the horizontal line
        for i in reversed(range(0, self.grid_layout.count())): # Start at 2 to keep the headers
            widget = self.grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

class FitPointsTable(ExportTableWidget):
    def __init__(self, parent=None):
        super().__init__()

    def fill_widget(self, fitted_data):
        
        # Check if the fitted data is a dictionary or a pd.DataFrame
        if isinstance(fitted_data, dict):
            data = fitted_data
        elif isinstance(fitted_data, pd.DataFrame):
            data = fitted_data.to_dict()
        else:
            raise ValueError(f"Unsupported data type: {type(fitted_data)}")
        
        # Clear existing table
        self.clear()
        
        # Add headers
        data_keys = list(data.keys())
        self.setColumnCount(len(data_keys))
        self.setRowCount(len(data[data_keys[0]]))
        self.setHorizontalHeaderLabels(data_keys)
        
        # Add data
        for row_idx, row_key in enumerate(data_keys):
            row_data = data[row_key]
            for col_idx, value in enumerate(row_data):
                # get the exponent of the value
                value_magnitude = int(f'{value:.4e}'.split('e')[1])
                # if the magnitude is between -3 and 3, show 5 digit floats, otherwise show 4 digit floats in scientific notation
                if -3 <= value_magnitude <= 3:
                    item = QTableWidgetItem(f'{value:.5f}')
                else:
                    item = QTableWidgetItem(f'{value:.4e}')
                self.setItem(col_idx, row_idx, item)
                
        # Resize columns to fit content
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        
        self.show()

class FitReportTable(QWidget):
    def __init__(self):
        super().__init__()

        # Create a vertical layout to hold the results
        self.vertical_layout = QVBoxLayout()
        
        self.setLayout(self.vertical_layout)
        self.show()

    def clear(self):
        while self.vertical_layout.count():
            child = self.vertical_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def fill_widget(self, model_results):
        self.model_results = model_results

        # Clear existing rows except headers
        self.clear()
        
        # Add headers
        self.vertical_layout.addWidget(QLabel('Pretty Print'), 0)
        
        # Capture pretty print output
        pretty_print_output = StringIO()
        with contextlib.redirect_stdout(pretty_print_output):
            self.model_results.params.pretty_print(colwidth=12)
        pretty_print_text = pretty_print_output.getvalue()
        
        # Parse pretty print text into rows and columns
        lines = pretty_print_text.strip().split('\n')
        headers = lines[0].split()
        data = [line.split() for line in lines[1:]]
        
        # Create a QTableWidget to display the pretty print output
        self.table_widget = QTableWidget()
        self.table_widget.setRowCount(len(data))
        self.table_widget.setColumnCount(len(headers))
        self.table_widget.setHorizontalHeaderLabels(headers)

        # Set a smaller font for the table items
        font = self.table_widget.font()
        font.setPointSize(8)  # Adjust the font size as needed
        self.table_widget.setFont(font)

        for row_idx, row_data in enumerate(data):
            for col_idx, item in enumerate(row_data):
                self.table_widget.setItem(row_idx, col_idx, QTableWidgetItem(item))

        # Adjust the column widths
        self.table_widget.resizeColumnsToContents()
        for col in range(self.table_widget.columnCount()):
            current_width = self.table_widget.columnWidth(col)
            new_width = max(current_width - 20, 20)  # Ensure a minimum width
            self.table_widget.setColumnWidth(col, new_width)

        # Adjust the row heights
        self.table_widget.resizeRowsToContents()
        for row in range(self.table_widget.rowCount()):
            current_height = self.table_widget.rowHeight(row)
            new_height = max(current_height - 2, 20)  # Ensure a minimum height
            self.table_widget.setRowHeight(row, new_height)

        # Force the table widget to update and repaint to ensure changes take effect
        self.table_widget.resizeColumnsToContents()
        self.table_widget.update()
        
        # Apply stylesheet to the table header
        self.table_widget.horizontalHeader().setStyleSheet(
            "QHeaderView::section { background-color: rgb(45, 55, 65); \nborder-color: rgb(200, 200, 200);}\n")
        self.table_widget.verticalHeader().setStyleSheet(
            "QHeaderView::section { background-color: rgb(45, 55, 65); \nborder-color: rgb(200, 200, 200);}\n")
        self.vertical_layout.addWidget(self.table_widget, 3)
        
        # Add fit report
        self.vertical_layout.addWidget(QLabel('Fit Report'), 0)
        fit_report_text = self.model_results.fit_report()
        fit_report_widget = QTextEdit()
        fit_report_widget.setReadOnly(True)
        fit_report_widget.setText(fit_report_text)
        
        self.vertical_layout.addWidget(fit_report_widget, 3)
        self.table_widget.resizeColumnsToContents()
        self.table_widget.show()
        self.show()

    def clear(self):
        # Remove all widgets except the headers and the horizontal line
        for i in reversed(range(0, self.vertical_layout.count())):
            widget = self.vertical_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

class MyMplCanvas(FigureCanvas):
    def __init__(self, parent = None, Purpose = None):
        self.purpose = Purpose
        
        fig = Figure()
        #fig.tight_layout() #tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area
        self.axes = fig.add_subplot(111)
        super(MyMplCanvas, self).__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()
        self.figure.tight_layout()
        # Add the navigation toolbar
        self.toolbar = CustomNavigationToolbar(self, parent)
        
        self.axes.tick_params(axis='both', direction='in') # ticks inside the axes
        self.purpose_specific()
        # Create the cursor
        self.cursor = CustomCursor(self.axes, useblit=True, color='k', linestyle='-.', linewidth=1)

        # Create the selectors, but keep them inactive for now
        # self.selectors = {
        #     'rectangle': RectangleSelector(self.axes, self.on_select, useblit=True,
        #                                    button=[1, 3], # Don't use middle button
        #                                    interactive=True),
        #     'ellipse': EllipseSelector(self.axes, self.on_select, useblit=True,
        #                                button=[1, 3],  # Don't use middle button
        #                                interactive=True),
        #     'polygon': PolygonSelector(self.axes, self.on_select, useblit=True,
        #                                #props = dict(color='k', linestyle='-', linewidth=2, alpha=0.5, facecolor='darkcyan')
        #                                 ),
        #     'lasso': LassoSelector(self.axes, self.on_select, useblit=True,
        #                            button=[1, 3])  # Don't use middle button
        # }
        # for selector in self.selectors.values():
        #     selector.set_active(False)
            
        self.selectors = {}
        
            
        # # Connect the onmove method to the motion_notify_event
        # self.figure.canvas.mpl_connect('motion_notify_event', self.onmove)

    def setSelector(self, selector_type):
        # Example implementation of setting a selector
        self.selectors[selector_type] = CustomSelector(selector_type)

    # purpose specific function which is called in the __init__ function and after the clear function
    # determines small differences depending on the purpose of the canvas
    def purpose_specific(self):
        if self.purpose == "Hysteresis":
            # Draw horizontal and vertical lines through the origin in dark grey
            self.axes.axhline(0, color='darkgrey', zorder=0)
            self.axes.axvline(0, color='darkgrey', zorder=0)
            self.axes.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
            self.axes.set_xlim([-1, 1])
            self.axes.set_ylim([-1, 1])
            
            # Choose a color list for the hysteresis loops. Every item in this list should be a 2-tuple with the color for the positive and negative loop
            # The colors should be quite similar to each other with the second branch always being a bit darker than the first one
            self.color_list = [
                ("#99CCFF", "#6699CC"),  # Light blue, Darker blue
                ("#99FF99", "#66CC66"),  # Light green, Darker green
                ("#CC99FF", "#9966CC"),  # Light purple, Darker purple
                ("#FF9966", "#CC6633"),  # Light coral, Darker coral
                ("#66FF66", "#33CC33"),  # Light lime, Darker lime
                ("#FF9999", "#CC6666"),  # Light red, Darker red
                ("#FFCC99", "#CC9966"),  # Light orange, Darker orange
                ("#FF99FF", "#CC66CC"),  # Light pink, Darker pink
                ("#FFFF99", "#CCCC66"),  # Light yellow, Darker yellow
                ("#99FFFF", "#66CCCC"),  # Light cyan, Darker cyan
            ]
            
            # Flatten the color list and set it as the default color cycle
            flat_color_list = [color[0] for color in self.color_list]
            self.axes.set_prop_cycle(cycler('color', flat_color_list))
            
        elif self.purpose == "AFM":
            pass
        
        elif self.purpose == "Image":
            self.axes.axis('off')
        
        elif self.purpose == None:
            # Draw horizontal and vertical lines through the origin in dark grey
            self.axes.axhline(0, color='darkgrey', zorder=0)
            self.axes.axvline(0, color='darkgrey', zorder=0)
            
    # simple clear function which reinitializes the canvas afterwards with the purpose_specific function
    def clear(self):
        self.axes.clear()
        self.purpose_specific()
        self.figure.tight_layout()
        self.draw()
        
    # simple set_labels function which sets the labels of the canvas simultaneously for both axes
    def set_labels(self, xlabel, ylabel):
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.draw()

    def setSelector(self, selector_type):
        # Deactivate the current selector and activate the new one
        for selector in self.selectors.values():
            selector.set_active(False)
        self.selectors[selector_type].set_active(True)

    def on_select(self, arg1, arg2=None):
        if arg2 is None:
            # This is a LassoSelector callback
            # arg1 is a list of (x, y) pairs defining the vertices of the selection
            print('Vertices:', arg1)
        else:
            # This is a RectangleSelector or EllipseSelector callback
            # arg1 and arg2 are the press and release events
            print('Start position: (%f, %f)' % (arg1.xdata, arg1.ydata))
            print('End position: (%f, %f)' % (arg2.xdata, arg2.ydata))
            
    # def onmove(self, event):
    #     # Call the cursor's onmove method
    #     self.cursor.onmove(event)

    #     # Call the active selector's onmove method
    #     for selector in self.selectors.values():
    #         if selector.active:
    #             selector.onmove(event)

class JumpSlider(QSlider):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def hitButton(self, pos: QPoint) -> bool:
        # This function is called when the user clicks on the slider. It is necessary to use mousePressEvent everywhere on the slider widget and not just on the handle
        return self.rect().contains(pos)
    
    def mousePressEvent(self, event):
        # this is the magic: get a 0 - 1.0 value representing where the click occurred
        # (0.0 is at the left-most, 1.0 is at the right-most)
        ratio = event.pos().x() / self.width()
        value = round(ratio * (self.maximum() - self.minimum()))
        self.setValue(value)
        # it's important to call the superclass afterwards, otherwise the
        # slider handle won't move until you drag it
        super().mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        # change the paint color to white
        painter.setPen(Qt.GlobalColor.white)
        fm = QFontMetrics(painter.font())
        min_val = str(self.minimum())
        max_val = str(self.maximum())

        min_width = fm.horizontalAdvance(min_val)
        max_width = fm.horizontalAdvance(max_val)

        painter.drawText(self.rect().x() + 2, self.rect().y() + int(2.6 * fm.height()), min_val)
        painter.drawText(self.rect().x() + self.rect().width() - max_width, self.rect().y() + int(2.6 * fm.height()), max_val)
        

class ConsoleWidget(QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        
        self.console = RichJupyterWidget() # Create a console widget
        self.console.kernel_manager = QtInProcessKernelManager() # Create a kernel manager for the console
        self.console.kernel_manager.start_kernel() # Start the kernel
        self.console.kernel_manager.kernel.gui = "qt" # Set the GUI of the kernel to qt
        self.console.kernel_client = self.console.kernel_manager.client() # Create a kernel client for the console
        self.console.setParent(self) # Set the parent of the console to the EvaluixConsole widget
        self.console.kernel_client.start_channels()  # Start communication channel with kernel
        self.console.show() # Show the console widget
    
class EvaluixConsole(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # Create a QTextEdit to display the output of the IPython shell
        self.text_edit = QTextEdit(self)

        # Create a QPushButton to start the IPython shell
        self.button = QPushButton("Start IPython Shell", self)
        self.button.clicked.connect(self.start_shell)

        # Add the QTextEdit and QPushButton to the layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.button)

    def start_shell(self):
        def run_shell():
            try:
                import IPython
            except ImportError:
                print("IPython is not installed or not available in the system's PATH.")
            
            # Start the IPython shell in a separate process
            self.process = subprocess.Popen(["ipython"], stdout=subprocess.PIPE)

            # Read the output of the IPython shell and display it in the QTextEdit
            output = self.process.stdout.read()
            self.text_edit.append(output.decode())

        # Start the run_shell function in a separate thread
        threading.Thread(target=run_shell).start()

class HDF5PreviewDialog(QDialog):
    
    def __init__(self, filename):
        super().__init__()
        # Load the UI file
        uic.loadUi(hdf5preview_path, self)
        # Set the dialog to be deleted when closed
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        if filename:
            self.file = h5py.File(filename, 'r')
            self.view_hdf5()
        else:
            filename, _ = self.open_file_dialog()
        
        self.hdf5preview_dialog_btnbox = self.findChild(QDialogButtonBox, 'hdf5preview_dialog_btnbox')
        openButton = self.hdf5preview_dialog_btnbox.button(QDialogButtonBox.StandardButton.Open)
        applyButton = self.hdf5preview_dialog_btnbox.button(QDialogButtonBox.StandardButton.Apply)
        cancelButton = self.hdf5preview_dialog_btnbox.button(QDialogButtonBox.StandardButton.Cancel)
        
        openButton.clicked.connect(self.open_file_dialog)
        applyButton.clicked.connect(self.accept)
        applyButton.clicked.connect(self.close)
        cancelButton.clicked.connect(self.reject)
        cancelButton.clicked.connect(self.close)

    def view_hdf5(self):
        try:
            
            # load the QLabels in the dialog for displaying the file name and the file path
            self.hdf5preview_filename = self.findChild(QLabel, 'hdf5preview_filename')
            self.hdf5preview_filepath = self.findChild(QLabel, 'hdf5preview_filepath')
            name = self.file.filename.split("/")[-1]
            path = self.file.filename.split(name)[0]
            
            self.hdf5preview_filename.setText('Filename: ' + name)
            self.hdf5preview_filepath.setText('Filepath: ' + path)
            
            # load the QTreeWidget in the dialog for displaying the groups and subgroups
            self.hdf5preview_tree = self.findChild(QTreeWidget, 'hdf5preview_tree')
            # load the QTableWidget in the dialog for displaying the content of a group or subgroup
            self.hdf5preview_table_data = self.findChild(QTableWidget, 'hdf5preview_table_data')
            # load the QTableWidget in the dialog for displaying the attributes of a group or subgroup
            self.hdf5preview_table_attr = self.findChild(QTableWidget, 'hdf5preview_table_attr')

            # Populate the QTreeWidget with the groups and subgroups from the HDF5 file
            self.populate_tree(self.file)

            # Connect the itemClicked signal of the QTreeWidget to a slot that updates the QTableWidget
            self.hdf5preview_tree.itemClicked.connect(self.update_hdf5content)

            # Show the dialog
            self.show()
            
        # if an error occurs, close the dialog but print the error message
        except Exception as e:
            print(f"Error: {e}")
            self.close()

    def populate_tree(self, group, parent_item=None):
        # If parent_item is None, this is the root group
        if parent_item is None:
            # Check if the group is the root group
            if group.name == '/':
                # Assign a custom name to the root group
                group_name = 'Root'
            parent_item = QTreeWidgetItem(self.hdf5preview_tree, [group.name.split('/')[-1]])
            self.hdf5preview_tree.expandItem(parent_item)
        else:
            parent_item = QTreeWidgetItem(parent_item, [group.name.split('/')[-1]])

        # Set the UserRole data for the group
        parent_item.setData(0, Qt.ItemDataRole.UserRole, group)

        # Iterate over the items in the group
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                # This is a subgroup, add it to the tree and recurse
                self.populate_tree(item, parent_item)
            else:
                # This is a dataset, add it to the tree
                dataset_item = QTreeWidgetItem(parent_item, [key])
                dataset_item.setData(0, Qt.ItemDataRole.UserRole, item)

    def update_hdf5content(self, item, column):
        # Get the HDF5 group or dataset corresponding to the clicked item
        group_or_dataset = item.data(column, Qt.ItemDataRole.UserRole)

        # Clear the table
        self.hdf5preview_table_data.setRowCount(0)
        self.hdf5preview_table_data.setColumnCount(0)
        self.hdf5preview_table_attr.setRowCount(0)
        self.hdf5preview_table_attr.setColumnCount(2)
        self.hdf5preview_table_attr.setHorizontalHeaderLabels(['Attribute', 'Value'])

        # If group_or_dataset is None, there's nothing more to do
        if group_or_dataset is None:
            print(item, column, "is None")
            return

        # If this is a dataset, display its data in the table
        if isinstance(group_or_dataset, h5py.Dataset):
            try:
                if len(group_or_dataset.shape) > 1:
                    # Set the row count and column count to match the dataset's shape
                    self.hdf5preview_table_data.setRowCount(group_or_dataset.shape[0])
                    self.hdf5preview_table_data.setColumnCount(group_or_dataset.shape[1])

                    # If the dataset has a 'columns' attribute, use it as the horizontal headers
                    if 'columns' in group_or_dataset.attrs:
                        self.hdf5preview_table_data.setHorizontalHeaderLabels(group_or_dataset.attrs['columns'])

                    # Populate the table with the dataset's data
                    for row in range(group_or_dataset.shape[0]):
                        for col in range(group_or_dataset.shape[1]):
                            self.hdf5preview_table_data.setItem(row, col, QTableWidgetItem(str(group_or_dataset[row, col])))
                else:
                    # Set the row count and column count to match the dataset's shape
                    self.hdf5preview_table_data.setRowCount(group_or_dataset.shape[0])
                    self.hdf5preview_table_data.setColumnCount(1)
                    
                    for row in range(group_or_dataset.shape[0]):
                        self.hdf5preview_table_data.setItem(row, 0, QTableWidgetItem(str(group_or_dataset[row])))
                
            except ValueError:
                print(f"Error: {group_or_dataset} is not a valid dataset")
            except Exception as e:
                print(f"Error: {e}")
        
        # Display the group's attributes in the attributes table
        for row, (key, value) in enumerate(group_or_dataset.attrs.items()):
            self.hdf5preview_table_attr.insertRow(row)
            self.hdf5preview_table_attr.setItem(row, 0, QTableWidgetItem(key))
            self.hdf5preview_table_attr.setItem(row, 1, QTableWidgetItem(str(value)))
                
        self.hdf5preview_table_data.resizeColumnsToContents()
        self.hdf5preview_table_data.resizeRowsToContents()
        self.hdf5preview_table_attr.resizeColumnsToContents()
        self.hdf5preview_table_attr.resizeRowsToContents()
        
    def open_file_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", "", "HDF5 Files (*.h5)")
        if filename:
            self.file = h5py.File(filename, 'r')
            self.view_hdf5()
        else:
            self.close()
            
    def closeEvent(self, event):
        self.file.close()
        super().closeEvent(event)
        
class FunctionViewer(QDialog):
    applied = pyqtSignal(str, str, dict)
    
    def __init__(self):
        super().__init__()
        uic.loadUi(functionviewer_path, self)
        # Set the dialog to be deleted when closed
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        

        # Get the QDialogButtonBox from the UI file and assign the buttons
        self.functionviewer_dialog_btnbox = self.findChild(QDialogButtonBox, 'functionviewer_dialog_btnbox')
        applyButton = self.functionviewer_dialog_btnbox.button(QDialogButtonBox.StandardButton.Apply)
        cancelButton = self.functionviewer_dialog_btnbox.button(QDialogButtonBox.StandardButton.Cancel)
        
        applyButton.clicked.connect(self.apply_and_close)
        cancelButton.clicked.connect(self.reject)
        cancelButton.clicked.connect(self.close)

        # Create the QTreeWidget for the function names
        self.view_functions()

    def view_functions(self):
        try:
            # load the QTreeWidget in the dialog for displaying the groups and subgroups
            self.functionviewer_functree = self.findChild(QTreeWidget, 'functionviewer_functree')
            # load the QTableWidget in the dialog for displaying the content of a group or subgroup
            self.functionviewer_attr_assignment = self.findChild(QTableWidget, 'functionviewer_attr_assignment')
            self.functionviewer_attr_assignment.itemChanged.connect(self.resize_table)
            
            # load the QTableWidget in the dialog for displaying the attributes of a group or subgroup
            self.functionviewer_docstring = self.findChild(QTextEdit, 'functionviewer_docstring')
            self.functionviewer_docstring.setReadOnly(True)

            # Populate the QTreeWidget with the groups and subgroups from the HDF5 file
            self.populate_tree(["scipy", "numpy", "pandas", "EvaluationFunctions"])

            # Connect the itemClicked signal of the QTreeWidget to a slot that updates the QTableWidget
            self.functionviewer_functree.itemClicked.connect(self.on_item_clicked)

            # Show the dialog
            self.show()
            
        # if an error occurs, close the dialog but print the error message
        except Exception as e:
            print(f"Error: {e}" + traceback.format_exc())
            self.close()

    def populate_tree(self, modules):
        # Iterate over the list of module names
        for module_name in modules:
            # Import the module using importlib
            module = importlib.import_module(module_name)

            # Create a top-level item for the module
            module_item = QTreeWidgetItem(self.functionviewer_functree)
            module_item.setText(0, module_name)

            # Recursively add the members of the module to the tree
            self.add_module_members_to_tree(module, module_item)

        # Connect the itemClicked signal to a slot that updates the text edit
        self.functionviewer_functree.itemClicked.connect(self.on_item_clicked)

    def add_module_members_to_tree(self, module, parent_item, depth=0, max_depth=3):
        # Stop if the maximum depth is reached
        if depth > max_depth:
            return

        # Iterate over all members of the module
        for name, obj in inspect.getmembers(module):
            if name.startswith('_') or name.startswith('__'):
                continue
            # Check if the member is defined in the module
            if inspect.ismodule(obj):
                submodule_item = QTreeWidgetItem(parent_item)
                submodule_item.setText(0, name)

                # Recursively add the members of the submodule to the tree
                self.add_module_members_to_tree(obj, submodule_item, depth + 1)
            # If the member is a function, create a new tree item and add it to the parent item
            elif inspect.isfunction(obj):
                function_item = QTreeWidgetItem(parent_item)
                function_item.setText(0, name)

        # Expand the parent item so all its children are visible
        parent_item.setExpanded(False)

    def update_functiondetails(self, item, column):
        # Get the function name from the clicked item
        function_name = item.text(column)

        # Build the full module path by traversing up the tree
        module_path = []
        parent = item.parent()
        while parent is not None:
            module_path.append(parent.text(column))
            parent = parent.parent()
        module_path.reverse()

        # Join the module path components into a string
        module_name = ".".join(module_path)

        # Import the module
        if module_name:
            module = importlib.import_module(module_name)
        else:
            module = __import__('__main__')

        # Get the function object from the module
        function = getattr(module, function_name)
        
        self.update_signature(function)        
        self.update_docstring(function)

    def update_signature(self, function):
        # Get the signature of the function
        signature = inspect.signature(function)

        # Clear the QTableWidget
        self.functionviewer_attr_assignment.setRowCount(0)
        self.functionviewer_attr_assignment.setColumnCount(3)
        self.functionviewer_attr_assignment.setHorizontalHeaderLabels(['Parameter', 'Type', 'Default Value'])

        # Iterate over the parameters in the signature
        for name, param in signature.parameters.items():
            # Create a new row in the QTableWidget
            row = self.functionviewer_attr_assignment.rowCount()
            self.functionviewer_attr_assignment.insertRow(row)

            # Create a QTableWidgetItem for the parameter name and add it to the QTableWidget
            name_item = QTableWidgetItem(name)
            self.functionviewer_attr_assignment.setItem(row, 0, name_item)

            # Create a QTableWidgetItem for the parameter type and add it to the QTableWidget
            if param.annotation is not inspect._empty:
                type_item = QTableWidgetItem(str(param.annotation))
            else:
                type_item = QTableWidgetItem("see doc")
            self.functionviewer_attr_assignment.setItem(row, 1, type_item)

            # Create a QTableWidgetItem for the parameter default value and add it to the QTableWidget
            default = param.default if param.default is not param.empty else ""
            default_item = QTableWidgetItem(str(default))
            default_item.setFlags(default_item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.functionviewer_attr_assignment.setItem(row, 2, default_item)
            
        self.functionviewer_attr_assignment.resizeColumnsToContents()
        self.functionviewer_attr_assignment.resizeRowsToContents()    
        
    def resize_table(self):
        self.functionviewer_attr_assignment.resizeColumnsToContents()
        self.functionviewer_attr_assignment.resizeRowsToContents()
        
    def update_docstring(self, function):
        # Get the docstring of the function
        docstring = inspect.getdoc(function)

        # Set the docstring as the text of the text edit
        self.functionviewer_docstring.setText(docstring)
        
    def on_item_clicked(self, item, column):
        # Check if the item has any children
        if item.childCount() == 0:
            # If the item has no children, update the function details
            self.update_functiondetails(item, column)
        else:
            # If the item has children, toggle its expanded state
            item.setExpanded(not item.isExpanded())
        
    def apply_and_close(self):
        # Get the currently selected item
        item = self.functionviewer_functree.currentItem()

        # Get the function name from the selected item
        function_name = item.text(0)

        # Build the full module path by traversing up the tree
        module_path = []
        parent = item.parent()
        while parent is not None:
            module_path.append(parent.text(0))
            parent = parent.parent()
        module_path.reverse()

        # Join the module path components into a string
        module_name = ".".join(module_path)

        # Get the function parameters from the QTableWidget
        parameters = {}
        for row in range(self.functionviewer_attr_assignment.rowCount()):
            param_name = self.functionviewer_attr_assignment.item(row, 0).text()
            param_value = self.functionviewer_attr_assignment.item(row, 2).text()
            if param_value in ["", "None", "see doc"]:
                param_value = None
            parameters[param_name] = param_value

        # Emit the applied signal with the function and module names and parameters
        self.applied.emit(module_name, function_name, parameters)

        # Close the dialog
        self.close()
        
    def closeEvent(self, event):
        super().closeEvent(event) 
        
class ManualDataDialog(QDialog):
    applied = pyqtSignal(dict, dict)
    
    def __init__(self, parent=None, _id=None):
        super(QDialog, self).__init__(parent)

        # Load the UI from the .ui file
        uic.loadUi(manualdatadialog_path, self)
        
        # Set the dialog to be deleted when closed
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        
        # Get the ID from the global dataset
        crucial_metadata = ['internal_id', 'type', 'sample', 'user', 'device']
        metadata_guess = {'internal_id': _id, 'type': 'Hyst', 'sample': '2024_0001_1', 'user': 'ArVe', 'device': 'L-MOKE'}
        
        # Get the tables
        self.table_metadata = self.findChild(QTableWidget, 'mdi_table_metadata')
        self.table_raw_data = self.findChild(QTableWidget, 'mdi_table_rawdata')

        # Initialize tables with 2 rows and 10 columns
        self.table_metadata.setRowCount(2)
        self.table_metadata.setColumnCount(10)
        self.table_metadata.setVerticalHeaderLabels(['key', 'value'])
        for i in range(len(crucial_metadata)):
            self.table_metadata.setItem(0, i, QTableWidgetItem(crucial_metadata[i]))
            self.table_metadata.setItem(1, i, QTableWidgetItem(str(metadata_guess[crucial_metadata[i]])))
        
        self.table_raw_data.setRowCount(30)
        self.table_raw_data.setColumnCount(2)
        self.table_raw_data.setHorizontalHeaderLabels(['x', 'y1'])
        self.table_raw_data.setVerticalHeaderItem(0, QTableWidgetItem("column\nname"))
        self.table_raw_data.setItem(0, 0, QTableWidgetItem("field [mT]"))
        self.table_raw_data.setItem(0, 1, QTableWidgetItem("magnetization [a.u.]"))
        
        # Connect cellChanged signal to method
        self.table_metadata.cellChanged.connect(self.check_last_cell)
        self.table_raw_data.cellChanged.connect(self.check_last_cell)
        
        self.table_metadata.resizeColumnsToContents()
        self.table_metadata.resizeRowsToContents()
        self.table_raw_data.resizeColumnsToContents()
        self.table_raw_data.resizeRowsToContents()
        
        self.mdi_btnbox = self.findChild(QDialogButtonBox, 'mdi_btnbox')
        applyButton = self.mdi_btnbox.button(QDialogButtonBox.StandardButton.Apply)
        cancelButton = self.mdi_btnbox.button(QDialogButtonBox.StandardButton.Cancel)
        openButton = self.mdi_btnbox.button(QDialogButtonBox.StandardButton.Open)
        
        applyButton.clicked.connect(self.apply_and_close)
        cancelButton.clicked.connect(self.reject)
        cancelButton.clicked.connect(self.close)
        
        self.show()

    def check_last_cell(self, row, column):
        # Check if the last row or column was edited, # only for the raw data table
        if self.sender() == self.table_raw_data and row == self.sender().rowCount() - 1: 
            # Add a new row
            self.sender().insertRow(self.sender().rowCount())
            self.sender().setRowCount(self.sender().rowCount() + 1)
            # add a horzontal header label
            horizontal_headers = [self.sender().horizontalHeaderItem(i).text() for i in range(self.sender().columnCount())]
            horizontal_headers.append(f'y{self.sender().rowCount()}')
            self.sender().setHorizontalHeaderLabels(horizontal_headers)
            self.sender().resizeRowsToContents()
            self.sender().resizeColumnsToContents()
        if column == self.sender().columnCount() - 1:
            # Add a new column
            self.sender().insertColumn(self.sender().columnCount())
            self.sender().setColumnCount(self.sender().columnCount() + 1)
            # add a dummy info in the first row
            self.sender().setItem(0, self.sender().columnCount() - 1, QTableWidgetItem('Dummy'))
            self.sender().resizeRowsToContents()
            self.sender().resizeColumnsToContents()
                
    def keyPressEvent(self, event):
        # Check if the pressed key is Ctrl+V
        if event.matches(QKeySequence.StandardKey.Paste):
            # Get the clipboard contents
            clipboard = QGuiApplication.clipboard()
            text = clipboard.text()

            self.mdi_choosesep = self.findChild(QComboBox, 'mdi_choosesep')
            # Get the separator from the combobox
            separator = self.mdi_choosesep.currentText()
            
            # Split the text by the separator
            values = re.split(re.escape(separator), text)
            print(values, len(values), separator)
            try:
                # get the selected QTableWidget
                table = self.focusWidget()
                if not isinstance(table, QTableWidget):
                    print(f"Error: focusWidget() did not return a QTableWidget, got {type(table)} instead")
                    return

                # Get the currently selected cell
                start_row = table.currentRow()
                start_col = table.currentColumn()

                # Insert the values into the table
                for i, value in enumerate(values):
                    row = start_row + i // table.columnCount()
                    col = start_col + i % table.columnCount()

                    # If the row or column is out of range, add a new row or column
                    if row >= table.rowCount():
                        table.insertRow(table.rowCount())
                    if col >= table.columnCount():
                        table.insertColumn(table.columnCount())

                    # Create a new table widget item with the value and add it to the table
                    item = QTableWidgetItem(value)
                    table.setItem(row, col, item)
            except BaseException as e:
                print(f"Error: {e}")
        else:
            # If the pressed key is not Ctrl+V, call the superclass's keyPressEvent method
            super().keyPressEvent(event)
                
    def apply_and_close(self):
        # Get the metadata from the metadata table
        metadata = {}
        for row in range(self.table_metadata.rowCount()):
            key_item = self.table_metadata.item(row, 0)
            value_item = self.table_metadata.item(row, 1)

            # Skip the row if the key item is empty
            if not key_item or not key_item.text():
                continue

            key = key_item.text()
            
            if not key == 'Dummy':

                # Set the value to an empty string if the value item is empty
                value = value_item.text() if value_item and value_item.text() else ""

                metadata[key] = value

        # Get the raw data from the raw data table
        raw_data = {}
        # the keys are saved in the first row of the table
        keys = [self.table_raw_data.item(0, col).text() for col in range(self.table_raw_data.columnCount())]
        for row in range(1, self.table_raw_data.rowCount()):
            values = [self.table_raw_data.item(row, col).text() if self.table_raw_data.item(row, col) is not None else '0' for col in range(self.table_raw_data.columnCount())]
            
            # Skip the row if all items are None or '0'
            if all(value == '0' for value in values):
                continue

            # Replace '0' with actual 0 for numerical processing
            values = [0 if value == '0' else value for value in values]

            raw_data[row] = dict(zip(keys, values))

        # Drop the keys == 'Dummy' from the raw data
        if 'Dummy' in raw_data.keys():
            print('Dummy in raw_data. Will be deleted')
            raw_data = {key: value for key, value in raw_data.items() if 'Dummy' not in value.keys()}

        # Emit the applied signal with the metadata and raw data
        self.applied.emit(metadata, raw_data)

        # Close the dialog
        self.close()
                
    def closeEvent(self, event):
        super().closeEvent(event)