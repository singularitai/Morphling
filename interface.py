import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Mock.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!

import sys
import subprocess
import application_backend as ab
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    
    def __init__(self):

        self.app = QtWidgets.QApplication(sys.argv)
        self.MainWindow = QtWidgets.QMainWindow()
        self.setupUi(self.MainWindow)
        self.MainWindow.show()
        sys.exit(self.app.exec_())
        
        
    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(873, 663)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(9, 9, 841, 631))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.SuperResolution = QtWidgets.QCheckBox(self.gridLayoutWidget)
        self.SuperResolution.setChecked(True)
        self.SuperResolution.setObjectName("SuperResolution")
        self.gridLayout.addWidget(self.SuperResolution, 7, 0, 1, 1)
        self.SingleSpeaker = QtWidgets.QCheckBox(self.gridLayoutWidget)
        self.SingleSpeaker.setObjectName("SingleSpeaker")
        self.gridLayout.addWidget(self.SingleSpeaker, 5, 0, 1, 1)
        self.CustomFace = QtWidgets.QCheckBox(self.gridLayoutWidget)
        self.CustomFace.setObjectName("CustomFace")
        self.gridLayout.addWidget(self.CustomFace, 8, 0, 1, 1)
        self.AudioOnly = QtWidgets.QCheckBox(self.gridLayoutWidget)
        self.AudioOnly.setChecked(False)
        self.AudioOnly.setObjectName("AudioOnly")
        self.gridLayout.addWidget(self.AudioOnly, 3, 0, 1, 1)
        self.Generate = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.Generate.setObjectName("Generate")
        self.gridLayout.addWidget(self.Generate, 18, 0, 1, 1)
        #self.gridLayout.addWidget(self.Generate, 13, 0, 1, 1)
        self.SelectCustomFace = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.SelectCustomFace.setEnabled(False)
        self.SelectCustomFace.setCheckable(False)
        self.SelectCustomFace.setChecked(False)
        self.SelectCustomFace.setFlat(False)
        self.SelectCustomFace.setObjectName("SelectCustomFace")
        self.gridLayout.addWidget(self.SelectCustomFace, 9, 0, 1, 1)
        self.SmilingFace = QtWidgets.QCheckBox(self.gridLayoutWidget)
        self.SmilingFace.setObjectName("SmilingFace")
        self.gridLayout.addWidget(self.SmilingFace, 6, 0, 1, 1)
        #self.progressBar = QtWidgets.QProgressBar(self.gridLayoutWidget)
        #self.progressBar.setProperty("value", 24)
        #self.progressBar.setObjectName("progressBar")
        #self.gridLayout.addWidget(self.progressBar, 18, 0, 1, 1)
        self.tacotron2 = QtWidgets.QCheckBox(self.gridLayoutWidget)
        self.tacotron2.setObjectName("tacotron2")
        #self.gridLayout.addWidget(self.tacotron2, 18, 0, 1, 1)
        self.gridLayout.addWidget(self.tacotron2, 13, 0, 1, 1)
        self.LJSpeech = QtWidgets.QCheckBox(self.gridLayoutWidget)
        self.LJSpeech.setObjectName("LJSpeech")
        self.gridLayout.addWidget(self.LJSpeech, 4, 0, 1, 1)
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.gridLayoutWidget)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.gridLayout.addWidget(self.plainTextEdit, 1, 0, 1, 1)
        self.VoiceNumber = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.VoiceNumber.setEnabled(False)
        self.VoiceNumber.setMaximum(123)
        self.VoiceNumber.setObjectName("VoiceNumber")
        self.gridLayout.addWidget(self.VoiceNumber, 11, 0, 1, 1)
        self.SpecificVoice = QtWidgets.QCheckBox(self.gridLayoutWidget)
        self.SpecificVoice.setObjectName("SpecificVoice")
        self.gridLayout.addWidget(self.SpecificVoice, 10, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.CustomFace.clicked.connect(self.SelectCustomFace.show)
        self.SpecificVoice.clicked.connect(self.VoiceNumber.show)
        
        self.Generate.clicked.connect(self.generate_clicked)
        self.SpecificVoice.clicked.connect(self.custom_voice_toggle)
        self.CustomFace.clicked.connect(self.custom_face_toggle)
        self.SelectCustomFace.clicked.connect(self.custom_face_clicked)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.custom_face_path = ""
        
    def custom_voice_toggle(self):
        
        if self.SpecificVoice.isChecked():
            self.VoiceNumber.setEnabled(True)
        else:
            self.VoiceNumber.setEnabled(False)
            
    def custom_face_toggle(self):
        
        if self.CustomFace.isChecked():
            self.SelectCustomFace.setEnabled(True)
            self.SelectCustomFace.setCheckable(True)
            self.SelectCustomFace.setChecked(True)
            self.SelectCustomFace.setFlat(True)
        else:
            self.SelectCustomFace.setEnabled(False)
            self.SelectCustomFace.setCheckable(False)
            self.SelectCustomFace.setChecked(False)
            self.SelectCustomFace.setFlat(False)
    
    def custom_face_clicked(self):
        import easygui
        import os
        path = easygui.fileopenbox()
        cwd = os.getcwd()
        if type(path) != type(None):
            relative_path = os.path.relpath(path, cwd)

            self.custom_face_path = relative_path
            print(relative_path)



    def generate_clicked(self):

        text = self.plainTextEdit.toPlainText().replace("\n", ".")
        
        mode = 0
        
        audio_only = bool(self.AudioOnly.isChecked())

        audio_model = 1
        if(self.LJSpeech.isChecked()):
            audio_model = 0

        if(self.tacotron2.isChecked()):
            audio_model = 2
        
        single_speaker = self.SingleSpeaker.isChecked()
        
        no_smiling = self.SmilingFace.isChecked()
        
        super_resolution = self.SuperResolution.isChecked()
        
        use_custom_face = self.CustomFace.isChecked()
        
        use_custom_voice = self.SpecificVoice.isChecked()
        custom_voice = self.VoiceNumber.value()
        command = ""
        command += "python application_backend -t {} ".format(text)
        
        command += "-m {} ".format(mode)
        command += "-ao {} ".format(audio_only)
        command += "-am {} ".format(audio_model)
        command += "-ss {} ".format(single_speaker)
        command += "-sm {} ".format(no_smiling)
        command += "-sr {} ".format(super_resolution)
        command += "-ucf {} ".format(use_custom_face)
        command += "-cf {} ".format(self.custom_face_path)
        command += "-ucv {} ".format(use_custom_voice)
        command += "-cv {}".format(custom_voice)

        print(command)

        ab.Generate(audio_model=audio_model, audio_only=audio_only, custom_face=self.custom_face_path, custom_voice=custom_voice, 
             full_text=text, mode=mode, single_speaker=single_speaker, smiling=no_smiling, super_resolution=super_resolution, 
             use_custom_face=use_custom_face, use_custom_voice=use_custom_voice)
        

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "SingularitAI | Morphling Tool"))
        self.SuperResolution.setText(_translate("MainWindow", "Apply Super Resolution"))
        self.SingleSpeaker.setText(_translate("MainWindow", "Single Speaker and Voice"))
        self.CustomFace.setText(_translate("MainWindow", "Use Custom Face"))
        self.AudioOnly.setText(_translate("MainWindow", "Generate Audio Only"))
        self.Generate.setText(_translate("MainWindow", "Generate"))
        self.SelectCustomFace.setText(_translate("MainWindow", "Select Custom Face"))
        self.SmilingFace.setText(_translate("MainWindow", "Avoid Smiling Faces"))
        self.LJSpeech.setText(_translate("MainWindow", "Use LJ Speech"))
        self.tacotron2.setText(_translate("MainWindow", "Use Tacotron Speech"))
        self.SpecificVoice.setText(_translate("MainWindow", "Use Specific Voice"))


if __name__ == "__main__":
    Ui_MainWindow()