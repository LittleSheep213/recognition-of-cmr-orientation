# Form implementation generated from reading ui file 'D:\workfile\pytorch_project\pro1\pyqt_class\my_first.ui'
#
# Created by: PyQt6 UI code generator 6.3.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_auto_adjust_tool(object):
    def setupUi(self, auto_adjust_tool):
        auto_adjust_tool.setObjectName("auto_adjust_tool")
        auto_adjust_tool.resize(1107, 612)
        auto_adjust_tool.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))
        self.centralWidget = QtWidgets.QWidget(auto_adjust_tool)
        self.centralWidget.setObjectName("centralWidget")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_2.setGeometry(QtCore.QRect(525, 242, 61, 41))
        self.pushButton_2.setStatusTip("")
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_ori = QtWidgets.QLabel(self.centralWidget)
        self.label_ori.setGeometry(QtCore.QRect(0, 0, 512, 512))
        self.label_ori.setFrameShape(QtWidgets.QFrame.Shape.Panel)
        self.label_ori.setTextFormat(QtCore.Qt.TextFormat.AutoText)
        self.label_ori.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_ori.setObjectName("label_ori")
        self.label_adj = QtWidgets.QLabel(self.centralWidget)
        self.label_adj.setGeometry(QtCore.QRect(600, 0, 512, 512))
        self.label_adj.setFrameShape(QtWidgets.QFrame.Shape.Panel)
        self.label_adj.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_adj.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.label_adj.setObjectName("label_adj")
        self.label_tip = QtWidgets.QLabel(self.centralWidget)
        self.label_tip.setGeometry(QtCore.QRect(130, 520, 231, 31))
        font = QtGui.QFont()
        font.setFamily("Berlin Sans FB Demi")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.label_tip.setFont(font)
        self.label_tip.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.label_tip.setText("")
        self.label_tip.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_tip.setObjectName("label_tip")
        self.label_rel = QtWidgets.QLabel(self.centralWidget)
        self.label_rel.setGeometry(QtCore.QRect(450, 520, 211, 41))
        self.label_rel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_rel.setObjectName("label_rel")
        self.label_tip_2 = QtWidgets.QLabel(self.centralWidget)
        self.label_tip_2.setGeometry(QtCore.QRect(750, 520, 231, 31))
        font = QtGui.QFont()
        font.setFamily("Berlin Sans FB Demi")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.label_tip_2.setFont(font)
        self.label_tip_2.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.label_tip_2.setText("")
        self.label_tip_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_tip_2.setObjectName("label_tip_2")
        auto_adjust_tool.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(auto_adjust_tool)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1107, 26))
        self.menuBar.setObjectName("menuBar")
        self.menu_O = QtWidgets.QMenu(self.menuBar)
        self.menu_O.setObjectName("menu_O")
        auto_adjust_tool.setMenuBar(self.menuBar)
        self.actionopen = QtGui.QAction(auto_adjust_tool)
        self.actionopen.setObjectName("actionopen")
        self.actionsave = QtGui.QAction(auto_adjust_tool)
        self.actionsave.setObjectName("actionsave")
        self.menu_O.addAction(self.actionopen)
        self.menu_O.addAction(self.actionsave)
        self.menuBar.addAction(self.menu_O.menuAction())

        self.retranslateUi(auto_adjust_tool)
        QtCore.QMetaObject.connectSlotsByName(auto_adjust_tool)

    def retranslateUi(self, auto_adjust_tool):
        _translate = QtCore.QCoreApplication.translate
        auto_adjust_tool.setWindowTitle(_translate("auto_adjust_tool", "Auto_Adjust_Tool"))
        self.pushButton_2.setToolTip(_translate("auto_adjust_tool", "点击自动校准"))
        self.pushButton_2.setText(_translate("auto_adjust_tool", "adjust"))
        self.label_ori.setText(_translate("auto_adjust_tool", "原图加载区域"))
        self.label_adj.setText(_translate("auto_adjust_tool", "调整后的图片加载区域"))
        self.label_rel.setText(_translate("auto_adjust_tool", "预测结果"))
        self.menu_O.setTitle(_translate("auto_adjust_tool", "File"))
        self.actionopen.setText(_translate("auto_adjust_tool", "open"))
        self.actionsave.setText(_translate("auto_adjust_tool", "save"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    auto_adjust_tool = QtWidgets.QMainWindow()
    ui = Ui_auto_adjust_tool()
    ui.setupUi(auto_adjust_tool)
    auto_adjust_tool.show()
    sys.exit(app.exec())