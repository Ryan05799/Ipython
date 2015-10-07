
# coding: utf-8

# In[2]:

from tkinter import *
from PIL import Image, ImageTk

root=Tk()

root.title("My Image")

w = Canvas(root, width=640, height=480)
image = Image.open("test.bmp")
w.create_image((640, 480), image=ImageTk.PhotoImage(image))

w.pack()

root.mainloop()


# In[3]:

image

