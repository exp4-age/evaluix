import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QComboBox, QLineEdit, QStyledItemDelegate
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure()
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.plot(1)  # Default dataset is 1
        self.plot(2)  # Default dataset is 2

    def plot(self, dataset):
        #self.ax.clear()
        if dataset == 1:
            self.line, = self.ax.plot([0, 1, 2, 3], [10, 1, 20, 3], 'r-', label='Line 1')
        else:
            self.line, = self.ax.plot([0, 1, 2, 3], [5, 15, 5, 15], 'b-', label='Line 2')
        self.ax.legend()
        self.draw()

    def update_plot(self, color=None, marker=None, linestyle=None, label=None):
        if color:
            self.line.set_color(color)
        if marker is not None:
            self.line.set_marker(marker if marker != "None" else '')
        if linestyle is not None:
            self.line.set_linestyle(linestyle if linestyle != "None" else '')
        if label:
            self.line.set_label(label)
            self.ax.legend()
        self.draw()

class ColorDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        painter.save()
        color_name = index.data()
        color = QColor(color_name) if color_name else QColor("black")
        rect = option.rect

        # Draw the color block
        painter.setBrush(color)
        painter.drawRect(rect.adjusted(2, 2, -rect.width() + 20, -2))

        # Draw the text
        painter.setPen(Qt.GlobalColor.black)
        painter.drawText(rect.adjusted(25, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, color_name if color_name else "black")

        painter.restore()

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        colors = ["red", "green", "blue", "yellow", "black", "white"]
        combo.addItems(colors)
        combo.showPopup()  # Open the dropdown immediately
        return combo

    def setEditorData(self, editor, index):
        value = index.data()
        editor.setCurrentText(value)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Matplotlib Plot Customizer")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.plot_canvas = PlotCanvas(self)
        self.layout.addWidget(self.plot_canvas)

        self.table_widget = QTableWidget(3, 10)
        self.table_widget.setHorizontalHeaderLabels(["Dataset", "Color", "Marker", "Line Style", "Label"])
        self.table_widget.verticalHeader().setVisible(False)
        self.layout.addWidget(self.table_widget)

        self.init_table()

    def init_table(self):
        properties = ["Dataset", "Color", "Marker", "Line Style", "Label"]
        values1 = ["1", "red", "None", "-", "Line 1"]
        values2 = ["2", "blue", "None", "-", "Line 2"]

        for col, (prop, val1, val2) in enumerate(zip(properties, values1, values2)):
            self.table_widget.setItem(0, col, QTableWidgetItem(prop))
            if prop == "Dataset":
                dataset_item1 = QTableWidgetItem(val1)
                dataset_item1.setFlags(Qt.ItemFlag.ItemIsEnabled)  # Make it non-editable
                self.table_widget.setItem(1, col, dataset_item1)
                
                dataset_item2 = QTableWidgetItem(val2)
                dataset_item2.setFlags(Qt.ItemFlag.ItemIsEnabled)  # Make it non-editable
                self.table_widget.setItem(2, col, dataset_item2)
            elif prop == "Color":
                color_item1 = QTableWidgetItem(val1)
                self.table_widget.setItem(1, col, color_item1)
                self.table_widget.setItemDelegateForColumn(1, ColorDelegate(self.table_widget))
                self.table_widget.itemChanged.connect(self.change_color)
                
                color_item2 = QTableWidgetItem(val2)
                self.table_widget.setItem(2, col, color_item2)
                self.table_widget.setItemDelegateForColumn(2, ColorDelegate(self.table_widget))
                self.table_widget.itemChanged.connect(self.change_color)
            elif prop == "Marker":
                marker_combo1 = QComboBox()
                marker_combo1.addItems(["None", "o", "s", "^", "D", "x", "+", "*"])
                marker_combo1.setCurrentText(val1)
                marker_combo1.currentTextChanged.connect(self.change_marker)
                self.table_widget.setCellWidget(1, col, marker_combo1)
                
                marker_combo2 = QComboBox()
                marker_combo2.addItems(["None", "o", "s", "^", "D", "x", "+", "*"])
                marker_combo2.setCurrentText(val2)
                marker_combo2.currentTextChanged.connect(self.change_marker)
                self.table_widget.setCellWidget(2, col, marker_combo2)
            elif prop == "Line Style":
                linestyle_combo1 = QComboBox()
                linestyle_combo1.addItems(["None", "-", "--", "-.", ":"])
                linestyle_combo1.setCurrentText(val1)
                linestyle_combo1.currentTextChanged.connect(self.change_linestyle)
                self.table_widget.setCellWidget(1, col, linestyle_combo1)
                
                linestyle_combo2 = QComboBox()
                linestyle_combo2.addItems(["None", "-", "--", "-.", ":"])
                linestyle_combo2.setCurrentText(val2)
                linestyle_combo2.currentTextChanged.connect(self.change_linestyle)
                self.table_widget.setCellWidget(2, col, linestyle_combo2)
            elif prop == "Label":
                label_edit1 = QLineEdit(val1)
                label_edit1.textChanged.connect(self.change_label)
                self.table_widget.setCellWidget(1, col, label_edit1)
                
                label_edit2 = QLineEdit(val2)
                label_edit2.textChanged.connect(self.change_label)
                self.table_widget.setCellWidget(2, col, label_edit2)

    def change_color(self, item):
        if item.column() == 1 and self.table_widget.item(item.row(), 0).text() == "Color":
            self.plot_canvas.ax.legend().get_lines()[0].set_color(item.text())
            self.plot_canvas.update_plot(color=item.text())

    def change_marker(self, marker):
        self.plot_canvas.update_plot(marker=marker)

    def change_linestyle(self, linestyle):
        self.plot_canvas.update_plot(linestyle=linestyle)

    def change_label(self, label):
        self.plot_canvas.update_plot(label=label)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    
