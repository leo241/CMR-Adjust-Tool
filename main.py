
def script_method(fn, _rcb=None):
    return fn


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj


import torch.jit
script_method1 = torch.jit.script_method
script1 = torch.jit.script
torch.jit.script_method = script_method
torch.jit.script = script


import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import numpy as np
import sys
from PyQt5.QtCore import QRect, QMetaObject, QCoreApplication, Qt, QSize
from PyQt5.QtGui import QPixmap, QIcon, QImage, QFont
from PyQt5.QtWidgets import *
from SimpleITK import GetArrayFromImage, ReadImage, WriteImage, GetImageFromArray
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import warnings

warnings.filterwarnings("ignore")  #ignore warnings

# 下面要把cnn这个类拿出来，否则无法加载模型
class cnn(nn.Module): # construction of netral network
    def __init__(self):
        super(cnn, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Sequential(
            nn.Conv2d( # 1 224 224
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2 # if stride = 1 padding = (kernel_size - 1)/2
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 16,128,128
        )
        # 16 224 224
        self.conv2 = nn.Sequential( # 16,128,128
            nn.Conv2d(16,32,5,1,2), # 32 128 128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 32 64 64
        )
        #
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,5,1,2),# 64 32 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 64 16 16
        )
        self.fc1 = nn.Linear(64*32*32, 64)
        self.out= nn.Linear(64, 8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        # print(x.size(), '进入全连接层前的维度')
        x = self.relu(self.fc1(x))
        x = self.out(x)
        return x




class auto:
    def __init__(self): # load 3 models
        self.directs = ["000", "001", "010", "011", "100", "101", "110", "111"]
        self.direct = ""

    def predict(self, img, model): # 改动：这里img只是 二维图像经过标准化之后延展成的四维的Tensor
        transform = transforms.Compose([  # transform to figure, for further passing to nn
            transforms.Resize((256, 256)),
            transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
        ])

        if len(img.shape) == 2: # 如果是二维图像
            std = img.std()
            if std == 0.0:  # 这里是为了防止std为0
                std = 1
            img_st = (img - img.mean()) / std # 将图像像素点进行（0，1）标准化
            img_tensor = transform(Image.fromarray(img_st))  # 要变成Image类型才能后续用transform转换
            img_tensor4d = img_tensor.unsqueeze(0)
            predict = float(torch.max(model(img_tensor4d), 1).indices)  # 把标准化好的图片放进神经网络中进行预测
            self.direct = self.directs[int(predict)]

        else:  # 如果是三维图像
            # img = img[:, :, 0]  # 方法一：如果是png的三维图像，取第一个slice图像作为目标
            slice_num = img.shape[2] # 方法二：投票法
            list_img = list()
            for slice_id in range(slice_num):
                img_slice = img[:,:,slice_id]
                std = img_slice.std()
                if std == 0.0: # 这里是为了防止std为0
                    std = 1
                img_st = (img_slice - img_slice.mean()) / std  # 将图像像素点进行（0，1）标准化
                img_tensor = transform(Image.fromarray(img_st)).unsqueeze(0)  # 要变成Image类型才能后续用transform转换
                list_img.append(img_tensor)
            img_tensor4d = torch.cat(list_img,dim=0)
            predict = torch.max(model(img_tensor4d), 1).indices # 各个图片的分类预测
            predict_result = int(max(set(predict),key=list(predict).count)) # 根据投票结果选出最终方向
            self.direct = self.directs[predict_result]
        return self.direct

    def adjust(self, img): # 注意，这里的img是读取的3d图片文件，所以如果要处理二维png或者jpg，要先扩充维度
        target = img
        if self.direct == "":
            return 0, False
        if self.direct == "000":
            target = img  # 000 Target[x,y,z]=Source[x,y,z]
        if self.direct == "001":
            target = np.fliplr(img)  # 001 Target[x,y,z]=Source[sx-x,y,z]
        if self.direct == "010":
            target = np.flipud(img)  # 010 Target[x,y,z]=Source[x,sy-y,z]
        if self.direct == "011":
            target = np.flipud(np.fliplr(img))  # 011 Target[x,y,z]=Source[sx-x,sy-y,z]
        if self.direct == "100":
            target = img.transpose((1, 0, 2))  # 100 Target[x,y,z]=Source[y,x,z]
        if self.direct == "101":
            # 101 Target[x,y,z]=Source[sx-y,x,z] 110 Target[x,y,z]=Source[y,sy-x,z]
            # target = np.fliplr(img.transpose((1, 0, 2)))
            target = np.flipud(img.transpose((1, 0, 2)))
        if self.direct == "110":
            # 110 Target[x,y,z]=Source[y,sy-x,z] 101 Target[x,y,z]=Source[sx-y,x,z]
            # target = np.flipud(img.transpose((1, 0, 2)))
            target = np.fliplr(img.transpose((1, 0, 2)))
        if self.direct == "111":
            target = np.flipud(np.fliplr(img.transpose((1, 0, 2))))  # 111 Target[x,y,z]=Source[sx-y,sy-x,z]
        return target, True

class Ui_MainWindow(QMainWindow):
    def __init__(self): # 初始化设置照搬
        super(Ui_MainWindow, self).__init__()
        self.centralwidget = QWidget(self)
        splash = QSplashScreen(QPixmap('./cover.png'))
        splash.show()
        self.setUpParams() # 设置参数
        self.setupUi()  # 设置界面

    def setupUi(self): # 页面构造照搬
        self.setObjectName("MainWindow")
        self.setGeometry(300, 100, 1200, 906)
        self.centralwidget.setObjectName("centralwidget")
        self.setWindowIcon(QIcon('./exeicon.ico'))

        self.statusBar()
        self.verticalLayoutWidget = QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QRect(250, 100, 650, 500))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QLabel(self.verticalLayoutWidget)
        size = QSize(680, 420)
        self.label.setPixmap(QPixmap.fromImage(QImage('./cover.png').scaled(size, Qt.IgnoreAspectRatio)))
        self.verticalLayout.addWidget(self.label)

        self.verticalLayoutWidget_8 = QWidget(self.centralwidget)
        self.verticalLayoutWidget_8.setGeometry(QRect(900, 200, 300, 500))
        self.verticalLayoutWidget_8.setObjectName("verticalLayoutWidget_8")
        self.verticalLayout_8 = QVBoxLayout(self.verticalLayoutWidget_8)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setObjectName("verticalLayout")
        self.label_instruction = QLabel(self.verticalLayoutWidget_8)
        size = QSize(300, 500)
        self.label_instruction.setPixmap(
            QPixmap.fromImage(QImage('./instruction.png').scaled(size, Qt.IgnoreAspectRatio)))
        self.verticalLayout_8.addWidget(self.label_instruction)

        self.verticalLayoutWidget_2 = QWidget(self.centralwidget)
        # self.verticalLayoutWidget_2.setGeometry(QRect(300, 660, 100, 30))
        self.verticalLayoutWidget_2.setGeometry(QRect(70, 270, 150, 45))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton = QPushButton(self.verticalLayoutWidget_2)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(lambda: self.openfiles()) # openfiles:pushButton
        self.pushButton.setStyleSheet('QPushButton {color: #424345;}')
        # self.pushButton.setStyleSheet('QPushButton {background-color: #2ab8d0; border: none;  color: #ffffff;}')
        self.verticalLayout_2.addWidget(self.pushButton)

        self.verticalLayoutWidget_3 = QWidget(self.centralwidget)
        self.verticalLayoutWidget_3.setGeometry(QRect(250, 630, 650, 200))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_3 = QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")

        self.verticalLayoutWidget_tag = QWidget(self.centralwidget)
        self.verticalLayoutWidget_tag.setGeometry(QRect(250, 820, 600, 50))
        self.verticalLayoutWidget_tag.setObjectName("verticalLayoutWidget_tag")
        self.verticalLayout_tag = QVBoxLayout(self.verticalLayoutWidget_tag)
        self.verticalLayout_tag.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_tag.setObjectName("verticalLayout_tag")
        self.tag_Label = QLabel(self)
        self.tag_Label.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.tag_Label.setStyleSheet("color:#8a8e8f")
        self.tag_Label.setObjectName("")
        self.tag_Label.setText("By Zhang, Ke @ ZMIC.Fudan University (2.0 Version Edited By Liu, Jim)")
        self.tag_Label.setFixedSize(600, 50)
        self.verticalLayout_tag.addWidget(self.tag_Label)


        self.verticalLayoutWidget_4 = QWidget(self.centralwidget)
        # self.verticalLayoutWidget_4.setGeometry(QRect(300, 710, 100, 30))
        self.verticalLayoutWidget_4.setGeometry(QRect(70, 340, 150, 45))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.verticalLayout_4 = QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.adjustButton = QPushButton(self.verticalLayoutWidget_4)
        self.adjustButton.setObjectName("adjustButton")
        self.adjustButton.setStyleSheet('QPushButton {color: #424345;}')
        self.adjustButton.clicked.connect(lambda: self.adjust())
        self.verticalLayout_4.addWidget(self.adjustButton)

        self.verticalLayoutWidget_5 = QWidget(self.centralwidget)
        # self.verticalLayoutWidget_5.setGeometry(QRect(300, 760, 100, 30))
        self.verticalLayoutWidget_5.setGeometry(QRect(70, 410, 150, 45))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.verticalLayout_5 = QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.predictButton = QPushButton(self.verticalLayoutWidget_5)
        self.predictButton.setObjectName("predictButton")
        self.predictButton.setStyleSheet('QPushButton {color: #424345;}')
        self.predictButton.clicked.connect(lambda: self.predict())
        self.verticalLayout_5.addWidget(self.predictButton)

        self.verticalLayoutWidget_6 = QWidget(self.centralwidget)
        # self.verticalLayoutWidget_6.setGeometry(QRect(300, 710, 100, 30))
        self.verticalLayoutWidget_6.setGeometry(QRect(70, 480, 150, 45))
        self.verticalLayoutWidget_6.setObjectName("verticalLayoutWidget_6")
        self.verticalLayout_6 = QVBoxLayout(self.verticalLayoutWidget_6)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.saveChangeHeaderButton = QPushButton(self.verticalLayoutWidget_6)
        self.saveChangeHeaderButton.setObjectName("saveChangeHeaderButton")
        self.saveChangeHeaderButton.setStyleSheet('QPushButton {color: #424345;}')
        self.saveChangeHeaderButton.clicked.connect(lambda: self.saveAs(reply=True))
        self.verticalLayout_6.addWidget(self.saveChangeHeaderButton)

        self.verticalLayoutWidget_7 = QWidget(self.centralwidget)
        # self.verticalLayoutWidget_7.setGeometry(QRect(300, 710, 100, 30))
        self.verticalLayoutWidget_7.setGeometry(QRect(70, 550, 150, 45))
        self.verticalLayoutWidget_7.setObjectName("verticalLayoutWidget_7")
        self.verticalLayout_7 = QVBoxLayout(self.verticalLayoutWidget_7)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.saveChangeImageButton = QPushButton(self.verticalLayoutWidget_7)
        self.saveChangeImageButton.setObjectName("saveChangeImageButton")
        self.saveChangeImageButton.setStyleSheet('QPushButton {color: #424345;}')
        self.saveChangeImageButton.clicked.connect(lambda: self.saveAs(reply=False))
        self.verticalLayout_7.addWidget(self.saveChangeImageButton)

        self.tableWidget = QTableWidget(self.verticalLayoutWidget_3)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setRowCount(0)
        item = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        self.verticalLayout_3.addWidget(self.tableWidget)
        self.setCentralWidget(self.centralwidget)

        self.verticalLayoutWidget_pageDown = QWidget(self.centralwidget)
        self.verticalLayoutWidget_pageDown.setGeometry(QRect(325, 50, 50, 64))
        self.verticalLayoutWidget_pageDown.setObjectName("verticalLayoutWidget_pageDown")
        self.verticalLayout_pageDown = QVBoxLayout(self.verticalLayoutWidget_pageDown)
        self.verticalLayout_pageDown.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_pageDown.setObjectName("verticalLayout_pageDown")
        self.verticalLayoutWidget_pageDown.setAttribute(Qt.WA_TranslucentBackground)
        self.DownButton = QPushButton(self.verticalLayoutWidget_pageDown)
        self.DownButton.setObjectName("DownButton")
        self.DownButton.setFixedSize(50, 64)
        self.DownButton.setIconSize(QSize(40, 40))
        self.DownButton.setIcon(QIcon("./left.png")) # DownButton: 往左翻页的按钮
        self.DownButton.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.DownButton.clicked.connect(lambda: self.OnClickDown())

        self.verticalLayoutWidget_NoSlices = QWidget(self.centralwidget)
        self.verticalLayoutWidget_NoSlices.setGeometry(QRect(390, 50, 100, 64))
        self.verticalLayoutWidget_NoSlices.setObjectName("verticalLayoutWidget_NoSlices")
        self.verticalLayout_NoSlices = QVBoxLayout(self.verticalLayoutWidget_NoSlices)
        self.verticalLayout_NoSlices.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_NoSlices.setObjectName("verticalLayout_NoSlices")
        self.NoSlices = QLabel("No.Slices:")
        self.NoSlices.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.NoSlices.setStyleSheet("color:#8a8e8f")
        self.NoSlices.setObjectName("NoSlices")
        self.NoSlices.setFixedSize(100, 64)
        self.verticalLayout_NoSlices.addWidget(self.NoSlices)

        self.lineEdit = QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QRect(495, 67, 30, 30))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setText("0") # 当前页数的按钮: self.lineEdit
        self.lineEdit.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.lineEdit.setAttribute(Qt.WA_TranslucentBackground)
        self.lineEdit.setStyleSheet("background-color: rgba(255, 255, 255, 0);color:#8a8e8f")
        self.lineEdit.editingFinished.connect(lambda: self.skipTo())

        self.verticalLayoutWidget_totalCount = QWidget(self.centralwidget)
        self.verticalLayoutWidget_totalCount.setGeometry(QRect(530, 67, 40, 30))
        self.verticalLayoutWidget_totalCount.setObjectName("verticalLayoutWidget_totalCount")
        self.verticalLayout_totalCount = QVBoxLayout(self.verticalLayoutWidget_totalCount)
        self.verticalLayout_totalCount.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_totalCount.setObjectName("verticalLayout_totalCount")
        self.totalCount = QLabel("/0") # 总页数的按钮：self.totalCount
        self.totalCount.setStyleSheet("color:#8a8e8f")
        self.totalCount.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.totalCount.setObjectName("")
        self.totalCount.setFixedSize(40, 30)
        # self.totalCount.addAction(self.openfiles())
        self.verticalLayout_totalCount.addWidget(self.totalCount)

        self.verticalLayoutWidget_pageUp = QWidget(self.centralwidget)
        self.verticalLayoutWidget_pageUp.setGeometry(QRect(575, 50, 50, 64))
        self.verticalLayoutWidget_pageUp.setObjectName("verticalLayoutWidget_pageUp")
        self.verticalLayout_pageUp = QVBoxLayout(self.verticalLayoutWidget_pageUp)
        self.verticalLayout_pageUp.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_pageUp.setObjectName("verticalLayout_pageUp")
        self.verticalLayoutWidget_pageUp.setAttribute(Qt.WA_TranslucentBackground)
        self.UpButton = QPushButton(self.verticalLayoutWidget_pageUp)
        self.UpButton.setObjectName("UpButton")
        self.UpButton.setFixedSize(50, 64)
        self.UpButton.setIconSize(QSize(40, 40))
        self.UpButton.setIcon(QIcon("./right.png")) # 向后翻页的按钮
        self.UpButton.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.UpButton.clicked.connect(lambda: self.OnClickUp())

        self.verticalLayoutWidget_Class = QWidget(self.centralwidget)
        self.verticalLayoutWidget_Class.setGeometry(QRect(650, 67, 60, 30))
        self.verticalLayoutWidget_Class.setObjectName("verticalLayoutWidget_Class")
        self.verticalLayout_Class = QVBoxLayout(self.verticalLayoutWidget_Class)
        self.verticalLayout_Class.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_Class.setObjectName("verticalLayout_Class")
        self.ClassLabel = QLabel("Class:")
        self.ClassLabel.setStyleSheet("color:#8a8e8f")
        self.ClassLabel.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.ClassLabel.setObjectName("")
        self.ClassLabel.setFixedSize(60, 30)
        self.verticalLayout_Class.addWidget(self.ClassLabel)

        self.classItems = QComboBox(self.centralwidget)
        self.classItems.addItems(['C0', 'T2', 'LGE']) # 封装类别的combobox
        self.classItems.setStyleSheet("color:#8a8e8f")
        self.classItems.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.classItems.setGeometry(QRect(720, 67, 70, 30))
        self.classItems.activated.connect(self.setClass)

        self.SLabel = QLabel(self)
        self.SLabel.setGeometry(575, 120, 55, 30)
        self.SLabel.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.SLabel.setStyleSheet("color:#2ab8d0")
        self.SLabel.setObjectName("")
        self.SLabel.setFixedSize(30, 30)

        self.ILabel = QLabel(self)
        self.ILabel.setGeometry(575, 560, 55, 30)
        self.ILabel.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.ILabel.setStyleSheet("color:#2ab8d0")
        self.ILabel.setObjectName("")
        self.ILabel.setFixedSize(30, 30)

        self.RALabel = QLabel(self)
        self.RALabel.setGeometry(265, 350, 50, 30)
        self.RALabel.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.RALabel.setStyleSheet("color:#2ab8d0")
        self.RALabel.setObjectName("")
        self.RALabel.setFixedSize(50, 30)

        self.LPLabel = QLabel(self)
        self.LPLabel.setGeometry(850, 350, 50, 30)
        self.LPLabel.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.LPLabel.setStyleSheet("color:#2ab8d0")
        self.LPLabel.setObjectName("")
        self.LPLabel.setFixedSize(50, 30)

        self.nameLabel = QLabel(self)
        self.nameLabel.setGeometry(370, 15, 400, 40)
        self.nameLabel.setFont(QFont("Microsoft YaHei", 8, QFont.Bold))
        self.nameLabel.setStyleSheet("color:#2ab8d0")
        self.nameLabel.setObjectName("")
        self.nameLabel.setFixedSize(400, 30)
        self.nameLabel.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)

        self.statusbar = QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.retranslateUi(self)
        QMetaObject.connectSlotsByName(self)
        self.show()

    def setUpParams(self):
        self.model_C0 = torch.load(f'model_C0.pkl')  # 这里要看好本地模型的名称
        self.model_T2 = torch.load(f'model_T2.pkl')
        self.model_LGE = torch.load(f'model_LGE.pkl')
        self.model = self.model_C0
        self.openPath = ""
        self.imgIndex = 0
        self.imgDim = 1
        self.adjust_imgIndex = 0
        self.adjust_imgDim = 6
        self.img = np.zeros((1, 1, 1))
        self.adjust_img = np.zeros((1, 1, 1))
        self.direct = 1
        self.auto = auto() # auto_adjust2.py中的auto类
        self.name = "C0"
        self.adjusted = False
        self.isOpen = False
        self.predicted = False
        self.Orientation_predicted = ""

    def retranslateUi(self, MainWindow):
        _translate = QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AutoAdjust Tool 2.0"))
        self.pushButton.setText(_translate("MainWindow", "Open"))
        self.pushButton.setFont(QFont("Microsoft YaHei", 12))
        self.adjustButton.setText(_translate("MainWindow", "Adjust"))
        self.adjustButton.setFont(QFont("Microsoft YaHei", 12))
        self.predictButton.setText(_translate("MainWindow", "Predict"))
        self.predictButton.setFont(QFont("Microsoft YaHei", 12))
        self.saveChangeHeaderButton.setText(_translate("MainWindow", "Save(Change Header)"))
        self.saveChangeHeaderButton.setFont(QFont("Microsoft YaHei", 10))
        self.saveChangeImageButton.setText(_translate("MainWindow", "Save(Change Image)"))
        self.saveChangeImageButton.setFont(QFont("Microsoft YaHei", 10))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Width"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Height"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "No.Slices"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Orientation"))

    def setClass(self, index):
        try:
            name = self.classItems.itemText(index)
            if name == 'C0':
                self.model = self.model_C0
            if name == 'T2':
                self.model = self.model_T2
            if name == 'LGE':
                self.model = self.model_LGE
        except:
            QMessageBox.information(self, "Tip", "Model Changing Failed! Please check your operation!")


    def openfiles(self):
        try:
            imgName, imgType = QFileDialog.getOpenFileName(self, "", './',"Figure Files (*.nii.gz *.nii *.mha *.png *.jpg);; all Files (*.*)")
                                                           # (self, "", os.getcwd(),"nii.gz Files (*.nii.gz);;mha Files (*.mha);;nii Files (*.nii)",'nii文件 (*.nii)')
            if imgName == "":
                return
            if "T2" in imgName: # 也就是说 如果图片名称里面不包括模型类别，会自动延用之前的模型
                self.name = "T2"
                self.model = self.model_T2
                self.classItems.setCurrentIndex(1) # combo-box下拉框的第几个，这里用来负责控制图片类别
            if "LGE" in imgName:
                self.name = "LGE"
                self.model = self.model_LGE
                self.classItems.setCurrentIndex(2)
            if "C0" in imgName:
                self.name = "C0"
                self.model = self.model_C0
                self.classItems.setCurrentIndex(0)
            self.OpenPath = imgName
            itk_img = ReadImage(imgName)
            img = GetArrayFromImage(itk_img)
            self.spacing = itk_img.GetSpacing()
            self.direction = itk_img.GetDirection()
            # print("img:", self.openPath, "direction:", self.direction)
            self.origin = itk_img.GetOrigin()
            minDim = list(img.shape).index(min(img.shape))  # 图像最短的边在第几维
            # print(minDim)
            if minDim == 0: # 保证最短的边在第三个维度上
                self.img = np.zeros((img.shape[1], img.shape[2], min(img.shape))) # 先用零矩阵起到placeholder的作用
                for i in range(min(img.shape)):
                    self.img[:, :, i] = img[i, :, :]
            if minDim == 1:
                self.img = np.zeros((img.shape[0], img.shape[2], min(img.shape)))
                for i in range(min(img.shape)):
                    self.img[:, :, i] = img[:, i, :]
            if minDim == 2:
                self.img = img
            self.imgDim = self.img.shape[2] # 把三维图像最短边的像素长度，设置为图像的维度
            if self.imgDim >= 3:
                self.imgIndex = int(self.imgDim / 2 - 1) # imgIndex相当于最短边的中点位置（z/2），（x,y,z/2）最能代表这个立体图像
            else:
                self.imgIndex = int(self.imgDim / 2 - 0.1)
            plt.cla()
            self.fig, self.ax = plt.subplots()
            self.ax.axis('off')
            # print('图像index：',self.imgIndex)
            self.ax.imshow(self.img[:, :, self.imgIndex], interpolation='nearest', aspect='auto', cmap='gray')
            cavan = FigureCanvas(self.fig)
            for i in reversed(range(self.verticalLayout.count())):
                self.verticalLayout.itemAt(i).widget().setParent(None)
            self.verticalLayout.addWidget(cavan)
            self.tableWidget.insertRow(0)
            item = QTableWidgetItem(str(self.img.shape[0]))
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
            self.tableWidget.setItem(0, 0, item)
            item = QTableWidgetItem(str(self.img.shape[1]))
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
            self.tableWidget.setItem(0, 1, item)
            item = QTableWidgetItem(str(self.img.shape[2]))
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
            self.tableWidget.setItem(0, 2, item)
            if not self.isOpen:
                self.LPLabel.setText("L/P")
                self.RALabel.setText("R/A")
                self.ILabel.setText("I")
                self.SLabel.setText("S")

            self.nameLabel.setText(imgName.split("/")[-1])
            self.totalCount.setText("/" + str(self.imgDim))
            self.lineEdit.setText(str(self.imgIndex + 1))

            self.adjusted = False
            self.predicted = False
            self.isOpen = True
        except:
            QMessageBox.information(self, "Tip", "Open Failed! Please check your operation!")

    def orientation_adjust(self):
        try:
            direction = np.reshape(np.array(list(self.direction)), (3, 3))
            if self.Orientation_predicted == "000":
                direction = direction  # 000 Target[x,y,z]=Source[x,y,z]
            if self.Orientation_predicted == "001":
                direction = np.fliplr(direction)  # 001 Target[x,y,z]=Source[sx-x,y,z]
            if self.Orientation_predicted == "010":
                direction = np.flipud(direction)  # 010 Target[x,y,z]=Source[x,sy-y,z]
            if self.Orientation_predicted == "011":
                direction = np.flipud(np.fliplr(direction))  # 011 Target[x,y,z]=Source[sx-x,sy-y,z]
            if self.Orientation_predicted == "100":
                direction = direction.transpose((1, 0, 2))  # 100 Target[x,y,z]=Source[y,x,z]
            if self.Orientation_predicted == "101":
                # 101 Target[x,y,z]=Source[sx-y,x,z] 110 Target[x,y,z]=Source[y,sy-x,z]
                # target = np.fliplr(img.transpose((1, 0, 2)))
                direction = np.flipud(direction.transpose((1, 0, 2)))
            if self.Orientation_predicted == "110":
                # 110 Target[x,y,z]=Source[y,sy-x,z] 101 Target[x,y,z]=Source[sx-y,x,z]
                # target = np.flipud(img.transpose((1, 0, 2)))
                direction = np.fliplr(direction.transpose((1, 0, 2)))
            if self.Orientation_predicted == "111":
                direction = np.flipud(
                    np.fliplr(direction.transpose((1, 0, 2))))  # 111 Target[x,y,z]=Source[sx-y,sy-x,z]
            direction = tuple(np.reshape(direction, (9,)).tolist())
            return direction
        except Exception:
            QMessageBox.information(self, "Tip", "Orientation adjust Failed! Please check your input!")

    def saveAs(self, reply):
        # set files
        if self.isOpen:
            try:
                if self.adjusted or self.Orientation_predicted == "000":
                    savePath, imgType = QFileDialog.getSaveFileName(self, "", "untitled_" + self.name,
                                                                    filter="nii.gz Files (*.nii.gz);;mha Files (*.mha);;nii Files (*.nii);;All Files (*)")
                    save_img = np.zeros((self.adjust_img.shape[2], self.adjust_img.shape[0], self.adjust_img.shape[1]))
                    for i in range(self.adjust_img.shape[2]):
                        save_img[i, :, :] = self.adjust_img[:, :, i]
                    # reply = QMessageBox.information(self, "Tip", "If you need to adjust the direction?",
                    #                                 QMessageBox.Yes | QMessageBox.Yes, QMessageBox.No)
                    # if reply == QMessageBox.Yes:
                    if reply:
                        direction = self.orientation_adjust()
                    else:
                        direction = self.direction
                    img_save = GetImageFromArray(save_img)
                    img_save.SetDirection(direction)
                    img_save.SetOrigin(self.origin)
                    img_save.SetSpacing(self.spacing)
                    WriteImage(img_save, savePath)
                else:
                    QMessageBox.information(self, "Tip", "Please adjust file first:)")
            except:
                QMessageBox.information(self, "Tip", "Sorry, this type of figure not supported to save yet T-T")
        else:
            QMessageBox.information(self, "Tip", "Please open file first:)")

    def predict(self):
        try:
            if self.isOpen:
                self.Orientation_predicted = self.auto.predict(self.img, self.model)
                item = QTableWidgetItem(self.Orientation_predicted)
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
                self.tableWidget.setItem(0, 3, item)
                self.adjusted = False
                if self.Orientation_predicted == "000":
                    self.adjust_img = self.img
            else:
                QMessageBox.information(self, "Tip", "Please open file first:)")
        except:
            QMessageBox.information(self, "Tip", "Predict Failed! Please check your input!")


    def adjust(self):
        # print(self.adjusted)
        # print(self.predicted)
        # print(self.img)
        try:
            if self.isOpen:
                if not self.adjusted:
                    if not self.predicted:
                        self.predict()
                    self.adjust_img, predicted = self.auto.adjust(self.img)
                    if not predicted:
                        self.predict()
                        self.adjust_img, predicted = self.auto.adjust(self.img)
                    self.img = self.adjust_img
                    plt.cla()
                    self.fig, self.ax = plt.subplots()
                    self.ax.axis('off')
                    self.ax.imshow(self.img[:, :, self.imgIndex], interpolation='nearest',
                                   aspect='auto', cmap='gray')
                    cavan = FigureCanvas(self.fig)
                    for i in reversed(range(self.verticalLayout.count())):
                        self.verticalLayout.itemAt(i).widget().setParent(None)
                    self.verticalLayout.addWidget(cavan)
                    self.adjusted = True
                else:
                    pass
            else:
                QMessageBox.information(self, "Tip", "Please open file first:)")
        except:
            QMessageBox.information(self, "Tip", "Adjust Failed! Please check your input!")

    def skipTo(self):
        targetNo = int(self.lineEdit.text())
        if self.isOpen and 0 < targetNo < self.imgDim + 1:
            self.ax.cla()
            self.ax.axis('off')
            self.ax.imshow(self.img[:, :, targetNo - 1], interpolation='nearest', aspect='auto',
                           cmap='gray')
            self.ax.figure.canvas.draw()

    def OnClickUp(self):
        try:
            if self.isOpen:
                try:
                    if self.imgIndex < self.imgDim - 1:
                        self.ax.cla()
                        self.imgIndex += 1
                        self.lineEdit.setText(str(self.imgIndex + 1))
                        self.ax.axis('off')
                        self.ax.imshow(self.img[:, :, self.imgIndex], interpolation='nearest',
                                       aspect='auto', cmap='gray')
                        self.ax.figure.canvas.draw()
                except IndexError:
                    pass
        except:
            pass

    def OnClickDown(self):
        try:
            if self.isOpen:
                try:
                    if self.imgIndex > 0:
                        self.ax.cla()
                        self.imgIndex -= 1
                        self.lineEdit.setText(str(self.imgIndex + 1))
                        self.ax.axis('off')
                        self.ax.imshow(self.img[:, :, self.imgIndex], interpolation='nearest',
                                   aspect='auto', cmap='gray')
                        self.ax.figure.canvas.draw()
                except IndexError:
                    pass
        except:
            pass



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Ui_MainWindow()
    ex.show()
    sys.exit(app.exec_())
