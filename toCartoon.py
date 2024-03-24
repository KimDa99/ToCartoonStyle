import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def SmoothOut(img):
    smooth = cv.medianBlur(img, 3)
    kernel_size = 11
    sigma_color = 30
    sigma_space = 3
    n_iterations = 2
    for itr in range(n_iterations):
        bil = cv.bilateralFilter(img, kernel_size, sigma_color, sigma_space)

    return bil

def ModifyColor(img):
    colorModified = img
    
    # Get HSV color space
    hsl_img = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    
    hue_channel = hsl_img[:,:,0]
    lightness_channel = hsl_img[:,:,1]
    saturation_channel = hsl_img[:,:,2]

    # equlize lightness
    equalized_lightness = cv.equalizeHist(lightness_channel)
    
    num_levels = 9  # Number of discrete levels
    discrete_lightness = np.round(equalized_lightness / (255 / (num_levels - 1))) * (255 / (num_levels - 1))
    
    discrete_lightness = discrete_lightness.astype(np.uint8)


    # equlize Saturation
    equalized_saturation = cv.equalizeHist(saturation_channel)
    equalized_saturation = saturation_channel

    num_levels = 5  # Number of discrete levels
    discrete_saturation = np.round(equalized_saturation / (255 / (num_levels - 1))) * (255 / (num_levels - 1))
    
    discrete_saturation = discrete_saturation.astype(np.uint8)

    # equlize Saturation
    num_levels = 20  # Number of discrete levels
    discrete_Hue = np.round(hue_channel / (179 / (num_levels - 1))) * (179 / (num_levels - 1))
    
    discrete_Hue = discrete_Hue.astype(np.uint8)



    # Merge back into HLS image
    modified_hls_img = np.dstack((discrete_Hue, discrete_lightness, discrete_saturation))
    
    # Convert HLS back to BGR
    colorModified = cv.cvtColor(modified_hls_img, cv.COLOR_HLS2BGR)
 
    return colorModified

def make_blue_black(img):
    # Convert image to HSV color space
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Define range of blue color in HSV
    lower_blue = np.array([90, 50, 50])    # Adjust these values as needed
    upper_blue = np.array([130, 255, 255]) # Adjust these values as needed

    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv_img, lower_blue, upper_blue)

    # Invert the mask (so non-blue regions are white)
    mask_inv = cv.bitwise_not(mask)

    # Set blue regions to black in the original image
    img[mask > 0] = [0, 0, 0]

    return img

def MakeOutline(original, img):

    kernel_size = 11
    sigma_color = 150
    sigma_space = 10
    n_iterations = 2

    for itr in range(n_iterations):
        bil = cv.bilateralFilter(original, kernel_size, sigma_color, sigma_space)
    # Compute gradient along x and y directions for each color channel
    gradient_x_b = cv.Sobel(bil[:,:,0], cv.CV_64F, 1, 0, ksize=kernel_size)
    gradient_y_b = cv.Sobel(bil[:,:,0], cv.CV_64F, 0, 1, ksize=kernel_size)
    gradient_x_g = cv.Sobel(bil[:,:,1], cv.CV_64F, 1, 0, ksize=kernel_size)
    gradient_y_g = cv.Sobel(bil[:,:,1], cv.CV_64F, 0, 1, ksize=kernel_size)
    gradient_x_r = cv.Sobel(bil[:,:,2], cv.CV_64F, 1, 0, ksize=kernel_size)
    gradient_y_r = cv.Sobel(bil[:,:,2], cv.CV_64F, 0, 1, ksize=kernel_size)

    # Compute magnitude of gradient for each color channel
    magnitude_gradient_b = np.sqrt(gradient_x_b**2 + gradient_y_b**2)
    magnitude_gradient_g = np.sqrt(gradient_x_g**2 + gradient_y_g**2)
    magnitude_gradient_r = np.sqrt(gradient_x_r**2 + gradient_y_r**2)

    # Normalize gradient magnitudes to [0, 1]
    normalized_magnitude_b = cv.normalize(magnitude_gradient_b, None, 0, 1, cv.NORM_MINMAX)
    normalized_magnitude_g = cv.normalize(magnitude_gradient_g, None, 0, 1, cv.NORM_MINMAX)
    normalized_magnitude_r = cv.normalize(magnitude_gradient_r, None, 0, 1, cv.NORM_MINMAX)

    # Combine the normalized gradient magnitudes
    combined_gradient = np.maximum(np.maximum(normalized_magnitude_b, normalized_magnitude_g), normalized_magnitude_r)

    threshold = 0.3
    # Threshold the combined gradient to find edges
    edge_mask = (combined_gradient > threshold).astype(np.uint8) * 255

    # Create a 3-channel outline image with edge mask applied to all channels
    outline = np.zeros_like(original)
    for c in range(outline.shape[2]):
        outline[:,:,c] = np.where(edge_mask > 0, original[:,:,c], 0)

    outLined = img.copy()
    outLined[edge_mask>0] = [0, 0, 0]

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

def GetCartoonRendered(img):
    smooth = SmoothOut(img)
    colorModifed = ModifyColor(smooth)
    outlined = MakeOutline(img, colorModifed)
    return outlined

def GetCartoonVideo(relativePath):
    video_file = relativePath

    # Read the given video
    video = cv.VideoCapture(video_file)

    # Check if the video is successfully opened
    if not video.isOpened():
        exit()
        print("Error: Could not open the video file.")

    # Get video properties
    fps = video.get(cv.CAP_PROP_FPS)
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    is_color = True  # Assuming the video is in color

    # Define the output video file name
    output_file = video_file[:video_file.rfind('.')] + '-cartoon.' + video_file[video_file.rfind('.') + 1:]

    # Initialize VideoWriter
    fourcc = int(video.get(cv.CAP_PROP_FOURCC))
    target = cv.VideoWriter(output_file, fourcc, fps, (width, height), is_color)

    # Check if the VideoWriter is successfully initialized
    if not target.isOpened():
        print("Error: Could not initialize the VideoWriter.")
        exit()

    # Process and write each frame to the output video
    while True:
        valid, frame = video.read()
        if not valid:
            break
        
        cartooned_frame = GetCartoonRendered(frame)
        
        # Write the processed frame to the output video
        target.write(cartooned_frame)
        
        # Display the processed frame (optional)
        cv.imshow('Cartoonized Frame', cartooned_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    # Release video objects
    video.release()
    target.release()
    cv.destroyAllWindows()    