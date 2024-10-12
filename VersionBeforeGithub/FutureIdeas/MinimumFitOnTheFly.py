import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QDialog, QComboBox, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt

from lmfit.models import LinearModel, QuadraticModel, PolynomialModel, GaussianModel, LorentzianModel, VoigtModel, PseudoVoigtModel, ExponentialModel, PowerLawModel, StepModel, RectangleModel, ExpressionModel
from lmfit import Model

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.datasets = []

    def plot(self, datasets):
        self.ax.clear()
        self.datasets = datasets
        for dataset in datasets:
            self.ax.plot(dataset['x'], dataset['y'], label=dataset.get('label', ''))
        self.ax.legend()
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI Selection Example")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.plot_canvas = PlotCanvas(self)
        self.layout.addWidget(self.plot_canvas)

        self.toolbar = NavigationToolbar(self.plot_canvas, self)
        self.layout.addWidget(self.toolbar)

        self.add_roi_button()

        self.selector = RectangleSelector(self.plot_canvas.ax, self.on_select,
                                          useblit=True,
                                          button=[1],  # Left mouse button
                                          minspanx=5, minspany=5,
                                          spancoords='pixels',
                                          interactive=True)
        self.selector.set_active(False)
        self.selected_points = []

    def add_roi_button(self):
        roi_button = QPushButton('Draw ROI', self)
        roi_button.clicked.connect(self.toggle_selector)
        self.toolbar.addWidget(roi_button)

    def toggle_selector(self):
        self.selector.set_active(True)

    def on_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # Define the ROI
        roi = [min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)]

        # Extract points within the ROI for all datasets
        self.selected_points = []
        for dataset in self.plot_canvas.datasets:
            selected_points = [(x, y) for x, y in zip(dataset['x'], dataset['y'])
                               if roi[0] <= x <= roi[1] and roi[2] <= y <= roi[3]]
            self.selected_points.append(selected_points)

        self.selector.set_active(False)
        self.open_fit_window()

    def open_fit_window(self):
        print("Selected points:", self.selected_points)
        fit_window = FitWindow(self.selected_points)
        fit_window.exec()

# class FitWindow(QDialog):
#     def __init__(self, selected_points):
#         super().__init__()
#         self.setWindowTitle("Fit Window")
#         self.selected_points = selected_points
#         self.init_ui()


#     def init_ui(self):
#         layout = QVBoxLayout()
#         self.setLayout(layout)
#         # Add UI elements to display or process selected points

class FitWindow(QDialog):
    def __init__(self, selected_points):
        super().__init__()
        self.setWindowTitle("Fit Data")
        self.selected_points = selected_points
        self.init_ui()

    def init_ui(self):
    
        # Create a dictionary of models
        self.models = {
            'linear': LinearModel,
            'quadratic': QuadraticModel,
            'polynomial': PolynomialModel,
            'gaussian': GaussianModel,
            'lorentzian': LorentzianModel,
            'voigt': VoigtModel,
            'pseudo_voigt': PseudoVoigtModel,
            'exponential': ExponentialModel,
            'power_law': PowerLawModel,
            'step': StepModel,
            'rectangle': RectangleModel,
            'expression': ExpressionModel
        }

        layout = QVBoxLayout()

        self.dataset_selector = QComboBox()
        self.dataset_selector.addItems(["Dataset 1", "Dataset 2"])
        layout.addWidget(self.dataset_selector)

        self.plot_canvas = FigureCanvas(Figure())
        self.ax = self.plot_canvas.figure.add_subplot(111)
        layout.addWidget(self.plot_canvas)

        self.dataset_selector.currentIndexChanged.connect(self.update_plot)
        self.update_plot()

        self.fit_type = QComboBox()
        self.fit_type.addItems(self.models.keys())
        layout.addWidget(self.fit_type)

        fit_button = QPushButton("Fit", self)
        fit_button.clicked.connect(self.perform_fit)
        layout.addWidget(fit_button)

        self.result_label = QLabel("")
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def update_plot(self):
        self.ax.clear()
        for points in self.selected_points:
            x, y = zip(*points)
            if points == self.selected_points[0]:
                self.points1 = points
                self.ax.plot(x, y, 'r.', label='Dataset 1')
            else:
                self.points2 = points
                self.ax.plot(x, y, 'b.', label='Dataset 2')
        self.ax.legend()
        self.plot_canvas.draw()

    def perform_fit(self):
        if self.dataset_selector.currentText() == "Dataset 1":
            points = self.points1
        else:
            points = self.points2

        x, y = zip(*points)
        x = np.array(x)
        y = np.array(y)
        
        model_name = self.fit_type.currentText().lower()
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' is not available.")
    
        model = self.models[model_name]()
        params = model.guess(y, x=x)
        result = model.fit(y, params, x=x)

        # Clear previous fit lines
        for line in self.ax.get_lines():
            if line.get_label() == 'fit':
                line.remove()

        # Plot new fit line
        self.ax.plot(x, result.best_fit, 'r-', label='fit')
        self.plot_canvas.draw()
        self.result_label.setText(f"Fit result: {result.fit_report()}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()

    x1 = np.linspace(0, 10, 100)
    y1 = 2 * x1 + 5 + np.random.normal(0, 2, 100)
    #x2 = np.linspace(0, 10, 100)
    #y2 = 3 * x2 + 7 + np.random.normal(0, 2, 100)
    x3 = np.linspace(0, 10, 100)
    # gaussian
    y3 = 0.5 * x3 + 3 + np.random.normal(0, 2, 100) + 20 * np.exp(-1 * ((x3 - 5) / 1) ** 2)

    # Example datasets
    datasets = [
        {'x': x1, 'y': y1, 'label': 'Dataset 1'},
        #{'x': x2, 'y': y2, 'label': 'Dataset 2'},
        {'x': x3, 'y': y3, 'label': 'Dataset 2'}
    ]
    main_window.plot_canvas.plot(datasets)

    sys.exit(app.exec())