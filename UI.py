# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'app.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MuseApp(object):
    def setupUi(self, MuseApp):
        MuseApp.setObjectName("MuseApp")
        MuseApp.resize(322, 262)
        self.centralwidget = QtWidgets.QWidget(MuseApp)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 10, 321, 241))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget = QtWidgets.QWidget(self.verticalLayoutWidget)
        self.widget.setObjectName("widget")
        self.verticalLayout.addWidget(self.widget)
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setMaximumSize(QtCore.QSize(16777215, 50))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(24)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.pushButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(10)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton, 0, QtCore.Qt.AlignHCenter)
        MuseApp.setCentralWidget(self.centralwidget)

        self.retranslateUi(MuseApp)
        QtCore.QMetaObject.connectSlotsByName(MuseApp)

    def retranslateUi(self, MuseApp):
        _translate = QtCore.QCoreApplication.translate
        MuseApp.setWindowTitle(_translate("MuseApp", "MainWindow"))
        self.label.setText(_translate("MuseApp", "TextLabel"))
        self.pushButton.setText(_translate("MuseApp", "start"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MuseApp = QtWidgets.QMainWindow()
    ui = Ui_MuseApp()
    ui.setupUi(MuseApp)
    MuseApp.show()
    sys.exit(app.exec_())