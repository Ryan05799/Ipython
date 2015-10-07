
# coding: utf-8

# In[82]:

import os
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk,ImageDraw


# In[83]:

class RoiMaker:
    
    filePath = './'
    black = (0, 0, 0)
    defaultWidth = 640
    defaultHeight = 480
    imgScale = 1.0
    
    def __init__(self, master):
        
        #Initialize attributes
        self.ptList = []
        self.lnList = []
        
        #Initialize widgets
        #Frames
        self.imgFrame = Frame(master, width = self.defaultWidth * 1 , height = self.defaultHeight)
        self.imgFrame.grid(row=0, column=0, padx=10, pady=10)
        self.resultFrame = Frame(master, width = self.defaultWidth/2 , height = self.defaultHeight)
        self.resultFrame.grid(row=0, column=1, padx=10, pady=10)
        self.btnFrame = Frame(master, width = 220, height = self.defaultHeight)
        self.btnFrame.grid(row=1, column=1, padx=10, pady=10)
        self.btnFrame.config(background = "#BBBBBB")  
        self.logFrame = Frame(master, width = 200 + self.defaultWidth, height = 100)
        self.logFrame.grid(row=1, column=0, padx=10, pady=10)
        #Canvas
        self.imgCanvas = Canvas(self.imgFrame, bd = 0, width = self.defaultWidth, height = self.defaultHeight)
        self.imgCanvas.grid(row=0, column=0)
        
        self.maskCanvas = Canvas(self.resultFrame, bd = 0, width = self.defaultWidth/2, height = self.defaultHeight/2)
        self.maskCanvas.grid(row=0, column=0)
        
        self.resultCanvas = Canvas(self.resultFrame, bd = 0, width = self.defaultWidth/2, height = self.defaultHeight/2)
        self.resultCanvas.grid(row=1, column=0)
       
        self.defaultImage = ImageTk.PhotoImage(Image.open("default.bmp"))        
        self.currentImg = self.imgCanvas.create_image(0,0,image = self.defaultImage ,anchor="nw")
        #binding image canvas to mouse click events
        self.imgCanvas.bind("<Button-1>", self.markCoord)
        
        #Buttons
        self.btnLoad = Button(self.btnFrame, height=3, width=12, text = "Load Image", command = self.loadImage)
        self.btnLoad.grid(row = 0, column = 0, padx = 10, pady = 30)
        
        self.btnApply = Button(self.btnFrame, height=3, width=12, text = "Apply", command = self.applyRoi)
        self.btnApply.grid(row = 0, column = 1, padx = 10, pady = 30)
        
        self.btnClear = Button(self.btnFrame, height=3, width=12, text = "Clear", command = self.clearRoi)
        self.btnClear.grid(row = 0, column = 2, padx = 10, pady = 30)
        
        self.btnSave = Button(self.btnFrame, height=3, width=12, text = "Save Mask", command = self.saveResult)
        self.btnSave.grid(row = 1, column = 0, padx = 10, pady = 30)
        
        self.btnRemove = Button(self.btnFrame, height=3, width=12, text = "Remove\n Reflectance", command = self.saveResult)
        self.btnRemove.grid(row = 1, column = 1, padx = 10, pady = 30)
        #Text
        self.logcat = Text(self.logFrame, width = 90, height = 10 )
        self.logcat.grid(row=0, column=0, padx=10, pady=20)
      
        
    
    #Binding Events for buttons, mouse click and pressing key  
    #Function for loading image file, called when btnLoad is click
    def loadImage(self):
        File = askopenfilename(parent=root, initialdir="./",title='Choose an image.')
        try:
            img = Image.open(File)
            self.imgScale = img.width/self.defaultWidth
            self.loadedImg = ImageTk.PhotoImage(img.resize(( int(self.defaultWidth) ,  
                                                            int(self.defaultHeight) ), Image.ANTIALIAS))
            self.imgCanvas.itemconfig(self.currentImg,image = self.loadedImg)   
        except:
            self.logcat.insert(0.0, 'Load image failed')
            return
        #record current image file path
        self.filePath = os.path.dirname(os.path.abspath(File))
        self.logcat.insert(0.0, 'Load image from ' + File + '\n')
        
        #clear previous ROI when new image is loaded
        self.clearRoi()
    
    #Applying the select points to forming the ROI region, and show results
    def applyRoi(self):
        self.logcat.insert(0.0, 'ROI applied\n')
        #draw the roi mask (of original image size)
        self.RoiMask = Image.new("L", (int(self.defaultWidth* self.imgScale ), int(self.defaultHeight * self.imgScale)), self.black)
        draw = ImageDraw.Draw(self.RoiMask)
        draw.polygon(self.ptList, fill="white")
        
        #display mask image on maskCanvas
        self.mskImg = ImageTk.PhotoImage(self.RoiMask.resize((int(self.defaultWidth/2), int(self.defaultHeight/2)), Image.ANTIALIAS))
        self.maskCanvas.create_image(0, 0, image = self.mskImg, anchor = 'nw')
        

    def clearRoi(self):
        for line in self.lnList:
            self.imgCanvas.delete(line)
        
        self.ptList = []
        self.lnList = []
        self.logcat.insert(0.0, 'Clear selected ROI\n')        
        
    #Output ROI mask
    def saveResult(self):
        filename = self.filePath + '/' + 'binMask.bmp'
        self.RoiMask.save(filename)
        self.logcat.insert(0.0, 'Save file to ' + filename + '\n')

    #Call the applyRoi function when enter key is pressed
    def enterForApply(self,event):
        self.applyRoi()
        
    #callback for marking the coordinates of clicked points
    def markCoord(self, event):
        
        #Map the canvas coordinates tooriginal image coordinates
        (x, y) = (int(event.x * self.imgScale), int(event.y * self.imgScale))
        
        #Add the point to the list and draw the line
        if len(self.ptList) > 0:
            #Recover the scale
            last_pt = (self.ptList[len(self.ptList) - 1][0]/self.imgScale, self.ptList[len(self.ptList) - 1][1]/self.imgScale)
            self.lnList.append(self.imgCanvas.create_line(int(last_pt[0]), int(last_pt[1]), event.x, event.y, fill  = "red"))
        self.ptList.append((x, y))
        self.logcat.insert(0.0, 'Mark (' + str(x) + ', ' + str(y) + ')\n')        
     
    #Function for remove the reflectance
    def rmReflect(self):
        print('Relectance region removed')


# In[84]:

root = Tk()
root.wm_title('ROI Select')
root.config(background = "#BBBBBB")  
roimaker = RoiMaker(root)
root.bind('<Return>', roimaker.enterForApply)
root.mainloop()

