import rawpy
import cv2
import numpy as np

# Set resize scale factor
scaleFactor = 0.2

# Set Image path in array
pathRawImage = [
    "raw image/raw image 1.NEF",
    "raw image/raw image 2.NEF",
    "raw image/raw image 3.NEF",
    "raw image/raw image 4.NEF",
]
pathGroundTruthImage = [
    "ground truth image/ground truth image 1.tiff",
    "ground truth image/ground truth image 2.tiff",
    "ground truth image/ground truth image 3.tiff",
    "ground truth image/ground truth image 4.tiff",
]

# Functions ***********************************************************************************************************


# Convert Raw to BGR ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def raw2bgr(imgSet, cameraWB=True):
    # Check if image is array or not
    if not isinstance(imgSet, list):
        return cv2.cvtColor(
            rawpy.imread(imgSet).postprocess(use_camera_wb=cameraWB), cv2.COLOR_RGB2BGR
        )
    else:
        return [
            cv2.cvtColor(
                rawpy.imread(img).postprocess(use_camera_wb=cameraWB), cv2.COLOR_RGB2BGR
            )
            for img in imgSet
        ]


# Convert RGB to Gray +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def rgb2gray(imgSet):
    # Check if image is array or not
    if not isinstance(imgSet, list):
        return cv2.cvtColor(imgSet, cv2.COLOR_RGB2GRAY)
    else:
        return [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgSet]


# Convert RGB to BGR ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def rgb2bgr(imgSet):
    # Check if image is array or not
    if not isinstance(imgSet, list):
        return cv2.cvtColor(imgSet, cv2.COLOR_RGB2BGR)
    else:
        return [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgSet]


# Convert BGR to Gray +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def bgr2gray(imgSet):
    # Check if image is array or not
    if not isinstance(imgSet, list):
        return cv2.cvtColor(imgSet, cv2.COLOR_BGR2GRAY)
    else:
        return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgSet]


# Convert BGR to RGB ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def bgr2rgb(imgSet):
    # Convert BGR image to RGB
    return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgSet]


# Convert Gray to BGR +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def gray2bgr(imgSet):
    # Check if image is array or not
    if not isinstance(imgSet, list):
        return cv2.cvtColor(imgSet, cv2.COLOR_GRAY2BGR)
    else:
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in imgSet]


# Read image set ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def readImage(imgSet):
    # Check if image is array or not
    if not isinstance(imgSet, list):
        return cv2.imread(imgSet)
    else:
        return [cv2.imread(img) for img in imgSet]


# Save image to TIFF ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def saveTIFF(imgSet, fileName):
    # Check if image is array or not
    if not isinstance(imgSet, list):
        cv2.imwrite(fileName, imgSet)
    elif len(imgSet) != len(fileName):
        print("Error: Image length mismatch")
    else:
        for i in range(len(imgSet)):
            cv2.imwrite(fileName[i], imgSet[i])
    return None


# Scale images ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def scaleImages(imgSet, scaleFactor=0.1):
    # Check if image is array or not
    if not isinstance(imgSet, list):
        return cv2.resize(imgSet, None, fx=scaleFactor, fy=scaleFactor)
    else:
        return [cv2.resize(img, None, fx=scaleFactor, fy=scaleFactor) for img in imgSet]


# Concatenate images, showing 2 images or more side by side or top to bottom ++++++++++++++++++++++++++++++++
def concatImages(imgSet1, imgSet2, axis=1):
    # Concatenate two images, axis=1 for horizontal, axis=0 for vertical
    # Check if imgSet is array or not
    if not isinstance(imgSet1, list) and not isinstance(imgSet2, list):
        return np.concatenate((imgSet1, imgSet2), axis=axis)
    elif not isinstance(imgSet1, list) or not isinstance(imgSet2, list):
        print("Error: Image type mismatch")
    elif len(imgSet1) != len(imgSet2):
        print("Error: Image length mismatch")
    else:
        # Concatenate images
        concatImg = [None] * len(imgSet1)
        for i in range(len(imgSet1)):
            # Check image size
            if imgSet1[i].shape[0] != imgSet2[i].shape[0]:
                print("Error: Image size mismatch at image " + str(i + 1))
                return None
            else:
                concatImg[i] = np.concatenate((imgSet1[i], imgSet2[i]), axis=axis)

        return concatImg
    return None


# Show all images +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def showImages(imgSet, name="image"):
    # Check if image is array or not
    if not isinstance(imgSet, list):
        cv2.imshow(name, imgSet)
        return None
    elif len(imgSet) == 0:
        print("Error: Image set is empty")
        return None
    else:
        for i in range(len(imgSet)):
            fileName = pathRawImage[i].split("/")[-1]
            cv2.imshow("Image " + str(i + 1) + ': "' + fileName + '"', imgSet[i])


# Main code ***********************************************************************************************************

# Step 1: Reading RAW Image and converting to BGR for OpenCV (Read RAW Image) ===============================
# Get Processed Image
rawImageSet = raw2bgr(pathRawImage)
# rawImageSet = raw2bgr("raw image/raw image 1.NEF")

# Step 2: Pre-process image, create "Ground Truth Image" (Grayscale, Resize, Save) ==========================
# Convert to Grayscale
# grayImageSet = bgr2gray(rawImageSet)
grayImageSet = bgr2gray(rawImageSet)

# Resize the image while maintaining the aspect ratio
grayResizedImagesSet = scaleImages(grayImageSet, scaleFactor)

# Save the processed image object to tagged image file TIFF as ground truth image
# saveTIFF(grayResizedImagesSet, pathGroundTruthImage)

# Step 3: Apply edge detection (Canny, Sobel, Laplacian, Prewitt, Scharr) ===================================
groundTruthImageSet = readImage(pathGroundTruthImage)



set1 = scaleImages(rawImageSet, scaleFactor)
set2 = scaleImages(gray2bgr(grayImageSet), scaleFactor)
rawXgroundTruthSet = concatImages(set1, set2)
# showImages(rawImageSet)
# showImages(groundTruthImageSet)
# showImages(grayImageSet)
# showImages(grayResizedImagesSet)
showImages(rawXgroundTruthSet, "Raw Image X Ground Truth Image")

cv2.waitKey(0)
cv2.destroyAllWindows()
