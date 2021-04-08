import os
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication,QWidget, QVBoxLayout, QPushButton, QFileDialog , QLabel, QTextEdit
import sys
from PyQt5.QtGui import QPixmap
from CnnImageClassifier import CnnImageClassifier 
from GradCamClass import GradCamClass
 
#GUI Window Class to browse through images and select the input image
class GUI(QWidget):
    imagePath=''
    model=0
    #Contructor to initialise the window class title and other parameters
    def __init__(self):
        super().__init__()
        #title of the input window
        self.title = "Image Classifier and object Localizer"
        #top variable initialization of window
        self.top = 400
        #top variable initialization of window
        self.left = 1000
        #top variable initialization of window
        self.width = 800
        #top variable initialization of window
        self.height = 600
        #initialise window
        self.InitWindow()
 
    #Method to Put widgets in initialised window pane
    def InitWindow(self):
        #Set window icon in the pane
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        #Set window title in the pane
        self.setWindowTitle(self.title)
        #Set window geometry in the pane
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        #creating Vbox layout object
        vbox = QVBoxLayout()
        self.label = QLabel("Select the image for classification")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        #adding label widget
        vbox.addWidget(self.label)
        #create the browsing button to select images for classification
        self.button1 = QPushButton("Browse Images")
        #trigger the getImage method in order to obtain the filepath and show image on the pane
        self.button1.clicked.connect(self.getImageAndPath)
        #create the process image button to submit image for classification
        self.button2 = QPushButton("Process Image")
        #trigger the process Image method in order to submit image for classification
        self.button2.clicked.connect(self.processImage)
        #adding button1 to the Vbox
        vbox.addWidget(self.button1)
        #adding button2 to the Vbox
        vbox.addWidget(self.button2)
        #setting label
        
        self.setLayout(vbox)
        self.show()
 
    #method to get ImagPath and display image on the window
    def getImageAndPath(self):
        #obtain the imagePath
        fname = QFileDialog.getOpenFileName(self, 'Open file','/Users/vikas/Documents/', "Image files (*.jpg *.gif *.png)")
        #save the path in global variable
        global imagePath
        #imagepath setting
        imagePath = fname[0]
        #make a pIxMap object to display image from imagePath
        pixmap = QPixmap(imagePath)
        # set the image in label for display
        self.label.setPixmap(QPixmap(pixmap))

    #call the Classifier to classify the image
    def processImage(self):
        print("processing image through trained CNN")
        GradCamClass.predictAndLocalizeObject(model,imagePath)

# Load trained CNN Model
model=CnnImageClassifier.loadTrainedCnnModel()

#Qapplication object creation        
App = QApplication(sys.argv)
#create a window class obbject
window = GUI()
sys.exit(App.exec())