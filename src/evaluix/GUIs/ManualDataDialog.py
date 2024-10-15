# Form implementation generated from reading ui file 'ManualDataDialog.ui'
#
# Created by: PyQt6 UI code generator 6.7.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_ManualDataInput_mdi(object):
    def setupUi(self, ManualDataInput_mdi):
        ManualDataInput_mdi.setObjectName("ManualDataInput_mdi")
        ManualDataInput_mdi.resize(1000, 840)
        ManualDataInput_mdi.setStyleSheet("background-color: rgb(25, 35, 45);\n"
"color: rgb(255, 255, 255);\n"
"border-color: rgb(200, 200, 200);\n"
"\n"
"")
        self.gridLayout_2 = QtWidgets.QGridLayout(ManualDataInput_mdi)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.mdi_choosesep = QtWidgets.QComboBox(parent=ManualDataInput_mdi)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.mdi_choosesep.sizePolicy().hasHeightForWidth())
        self.mdi_choosesep.setSizePolicy(sizePolicy)
        self.mdi_choosesep.setObjectName("mdi_choosesep")
        self.mdi_choosesep.addItem("")
        self.mdi_choosesep.addItem("")
        self.mdi_choosesep.addItem("")
        self.mdi_choosesep.addItem("")
        self.mdi_choosesep.addItem("")
        self.mdi_choosesep.addItem("")
        self.mdi_choosesep.addItem("")
        self.mdi_choosesep.addItem("")
        self.mdi_choosesep.addItem("")
        self.gridLayout_2.addWidget(self.mdi_choosesep, 4, 3, 1, 1)
        self.mdi_metadata_label = QtWidgets.QLabel(parent=ManualDataInput_mdi)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mdi_metadata_label.sizePolicy().hasHeightForWidth())
        self.mdi_metadata_label.setSizePolicy(sizePolicy)
        self.mdi_metadata_label.setObjectName("mdi_metadata_label")
        self.gridLayout_2.addWidget(self.mdi_metadata_label, 0, 1, 1, 1)
        self.mdi_table_metadata = QtWidgets.QTableWidget(parent=ManualDataInput_mdi)
        self.mdi_table_metadata.setStyleSheet("QHeaderView::section {\n"
"        background-color: rgb(45, 55, 65);\n"
"}")
        self.mdi_table_metadata.setObjectName("mdi_table_metadata")
        self.mdi_table_metadata.setColumnCount(0)
        self.mdi_table_metadata.setRowCount(0)
        self.gridLayout_2.addWidget(self.mdi_table_metadata, 1, 1, 1, 3)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 0, 2, 1, 2)
        self.mdi_rawdata_label = QtWidgets.QLabel(parent=ManualDataInput_mdi)
        self.mdi_rawdata_label.setObjectName("mdi_rawdata_label")
        self.gridLayout_2.addWidget(self.mdi_rawdata_label, 2, 1, 1, 1)
        self.mdi_btnbox = QtWidgets.QDialogButtonBox(parent=ManualDataInput_mdi)
        self.mdi_btnbox.setLocale(QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.Europe))
        self.mdi_btnbox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.mdi_btnbox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Apply|QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Open)
        self.mdi_btnbox.setObjectName("mdi_btnbox")
        self.gridLayout_2.addWidget(self.mdi_btnbox, 6, 0, 1, 3)
        self.mdi_label_sep = QtWidgets.QLabel(parent=ManualDataInput_mdi)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mdi_label_sep.sizePolicy().hasHeightForWidth())
        self.mdi_label_sep.setSizePolicy(sizePolicy)
        self.mdi_label_sep.setObjectName("mdi_label_sep")
        self.gridLayout_2.addWidget(self.mdi_label_sep, 3, 3, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 2, 2, 1, 2)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.gridLayout_2.addItem(spacerItem2, 5, 3, 1, 1)
        self.mdi_table_rawdata = QtWidgets.QTableWidget(parent=ManualDataInput_mdi)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(8)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.mdi_table_rawdata.sizePolicy().hasHeightForWidth())
        self.mdi_table_rawdata.setSizePolicy(sizePolicy)
        self.mdi_table_rawdata.setStyleSheet("QHeaderView::section {\n"
"        background-color: rgb(45, 55, 65);\n"
"}")
        self.mdi_table_rawdata.setObjectName("mdi_table_rawdata")
        self.mdi_table_rawdata.setColumnCount(0)
        self.mdi_table_rawdata.setRowCount(0)
        self.gridLayout_2.addWidget(self.mdi_table_rawdata, 3, 1, 3, 2)

        self.retranslateUi(ManualDataInput_mdi)
        self.mdi_btnbox.accepted.connect(ManualDataInput_mdi.accept) # type: ignore
        self.mdi_btnbox.rejected.connect(ManualDataInput_mdi.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(ManualDataInput_mdi)

    def retranslateUi(self, ManualDataInput_mdi):
        _translate = QtCore.QCoreApplication.translate
        ManualDataInput_mdi.setWindowTitle(_translate("ManualDataInput_mdi", "Dialog"))
        self.mdi_choosesep.setItemText(0, _translate("ManualDataInput_mdi", ","))
        self.mdi_choosesep.setItemText(1, _translate("ManualDataInput_mdi", "\\t"))
        self.mdi_choosesep.setItemText(2, _translate("ManualDataInput_mdi", "\\t+"))
        self.mdi_choosesep.setItemText(3, _translate("ManualDataInput_mdi", "\\s"))
        self.mdi_choosesep.setItemText(4, _translate("ManualDataInput_mdi", "\\s+"))
        self.mdi_choosesep.setItemText(5, _translate("ManualDataInput_mdi", ";"))
        self.mdi_choosesep.setItemText(6, _translate("ManualDataInput_mdi", "|"))
        self.mdi_choosesep.setItemText(7, _translate("ManualDataInput_mdi", ":"))
        self.mdi_choosesep.setItemText(8, _translate("ManualDataInput_mdi", "."))
        self.mdi_metadata_label.setText(_translate("ManualDataInput_mdi", "Metadata"))
        self.mdi_rawdata_label.setText(_translate("ManualDataInput_mdi", "Raw data"))
        self.mdi_label_sep.setText(_translate("ManualDataInput_mdi", "Seperator"))
