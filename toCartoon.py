import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def SmoothOut(img):
    smooth = img
    return smooth

def ModifyColor(img):
    colorModified = img
    # Get HSV color space
    #hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    return colorModified

def MakeOutline(img):
    outLined = img
    return outLined

def ShowHist(imgs):
    hists = []
    hists.append(GetHSVHist(imgs[0]))
    hists.append(GetHSVHist(imgs[2]))

    # Merge Images
    mergedImgs = np.hstack(imgs)
    mergedHists = np.vstack(hists)
    mergedHists = cv.resize(mergedHists, (max(mergedHists.shape[1], mergedImgs.shape[1]), mergedHists.shape[0]))
    mergedAll = np.vstack((mergedHists, mergedImgs))
    
    mergedAll = cv.resize(mergedAll,  (int(mergedAll.shape[1] * 0.5), int(mergedAll.shape[0] * 0.5)))

    cv.imshow('image', mergedAll)
    cv.waitKey(0)
    cv.destroyAllWindows()

def GetHSVHist(img):
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Extract the channels
    hue_channel = hsv_img[:,:,0]
    sat_channel = hsv_img[:,:,1]
    val_channel = hsv_img[:,:,2]
    
    # Calculate the histograms
    hue_hist = cv.calcHist([hue_channel], [0], None, [180], [0, 180])
    sat_hist = cv.calcHist([sat_channel], [0], None, [256], [0, 256])
    val_hist = cv.calcHist([val_channel], [0], None, [256], [0, 256])

    # Check if any of the histograms are empty (None)
    if hue_hist is None or sat_hist is None or val_hist is None:
        print("Error: Failed to generate histograms.")
        return None

    # Pad histograms with zeros if they have fewer bins than expected
    max_bins = max(hue_hist.shape[0], sat_hist.shape[0], val_hist.shape[0])
    hue_hist = np.pad(hue_hist, ((0, max_bins - hue_hist.shape[0]), (0, 0)), mode='constant')
    sat_hist = np.pad(sat_hist, ((0, max_bins - sat_hist.shape[0]), (0, 0)), mode='constant')
    val_hist = np.pad(val_hist, ((0, max_bins - val_hist.shape[0]), (0, 0)), mode='constant')

    # Plot the histograms
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(hue_hist, color='b')
    plt.title('Hue Histogram')
    plt.xlabel('Hue')
    plt.ylabel('Frequency')
    plt.xlim([0, 180])
    
    plt.subplot(1, 3, 2)
    plt.plot(sat_hist, color='g')
    plt.title('Saturation Histogram')
    plt.xlabel('Saturation')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    
    plt.subplot(1, 3, 3)
    plt.plot(val_hist, color='r')
    plt.title('Value (Brightness) Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])

    # returns Hist img
    #plt.tight_layout()
    #plt.show()

    # Convert plot to image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    hist_img = cv.imdecode(np.frombuffer(buf.getvalue(), np.uint8), 1)
    buf.close()
    plt.close()

    return hist_img



imgs = []
smooths = []
colorModifeds = []
cartoons = []

# get image
files = ['data/hill01.jpg', 'data/peppers.tif', 'data/blais.jpg', 'data/lena.tif']
for i in range(len(files)-1):
    imgs.append(cv.imread(files[i]))
    
    # smooth Out Image
    smooths.append(SmoothOut(imgs[i]))

    # modify color
    colorModifeds.append(ModifyColor(smooths[i]))

    # OutLine
    cartoons.append(colorModifeds[i])

    # Show Hist
    ShowHist([imgs[i], smooths[i], colorModifeds[i], cartoons[i]])