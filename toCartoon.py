import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def SmoothOut(img):
    return smooth

def ModifyColor(img):
    # Get HSV color space
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    return colorModified

def MakeOutline(img):
    return outLined

def ShowHist(imgs):
    for img in range(imgs):
        # Convert the image to HSV color space
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)


imgs = []
smooths = []
colorModifeds = []
cartoons = []

# get image
files = ['data/hill0.jpg', 'data/peppers.tif', 'data/blais.jpg', 'data/lena.tif']
for i in range(files):
    imgs.append(cv.imread(files[i]))
    
    # smooth Out Image
    smooths.append(SmoothOut(imgs[i]))

    # modify color
    colorModifeds.append(ModifyColor(smooths[i]))

    # OutLine
    cartoons.append(colorModifeds[i])

    # Show Hist
    ShowHist([imgs[i], smooths[i], colorModifeds[i], cartoons[i]])