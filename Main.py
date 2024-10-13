import func as f
import cv2

# Settings
SCALE_FACTOR = 0.0908

SAVE_NEW_IMAGE = False

# 1. Average filter
# 2. Box filter
# 3. Gaussian filter
# 4. Median filter
# 5. Bilateral filter
# 6. Non-local means filter
# 7. Custom kernel filter
SMOOTH_TYPE = 7
# Parameters for smoothing filter
SMOOTH_KERNEL_SIZE_X = 3
SMOOTH_KERNEL_SIZE_Y = 7
SMOOTH_KERNEL_SIZE = 5
SMOOTH_SIGMA_X = 0  # For gaussian filter
SMOOTH_SIGMA_Y = 0  # For gaussian filter
SMOOTH_D = 9  # For bilateral filter
SMOOTH_SIGMA_COLOR = 75  # For bilateral filter
SMOOTH_SIGMA_SPACE = 75  # For bilateral filter
SMOOTH_H = 10  # For non-local means filter
SMOOTH_SEARCH_WINDOW_SIZE = 20  # For non-local means filter

# 1. Canny
# 2. Sobel
# 3. Scharr
# 4. Laplacian
# 5. Prewitt
# 6. Roberts
# 7. Custom kernel filter
EDGEFILTERTYPE = 1

# Set Image path in array
IMG_NO = 3
pathRawImage = [
    "img/raw_v_1.NEF",
    "img/raw_v_2.NEF",
    "img/raw_v_3.NEF",
    "img/raw_v_4.NEF",
    "img/raw_h_1.NEF",
    "img/raw_h_2.NEF",
    "img/raw_h_3.NEF",
    "img/raw_h_4.NEF",
]

pathGTIGrayScaled = [
    "img/gti_gray_v_1.tiff",
    "img/gti_gray_v_2.tiff",
    "img/gti_gray_v_3.tiff",
    "img/gti_gray_v_4.tiff",
    "img/gti_gray_h_1.tiff",
    "img/gti_gray_h_2.tiff",
    "img/gti_gray_h_3.tiff",
    "img/gti_gray_h_4.tiff",
]

pathGTIRGBScaled = [
    "img/gti_rgb_v_1.tiff",
    "img/gti_rgb_v_2.tiff",
    "img/gti_rgb_v_3.tiff",
    "img/gti_rgb_v_4.tiff",
    "img/gti_rgb_h_1.tiff",
    "img/gti_rgb_h_2.tiff",
    "img/gti_rgb_h_3.tiff",
    "img/gti_rgb_h_4.tiff",
]

# Set Canny parameters
cannyParamSet = [
    {"tresh1": 100, "tresh2": 200, "apertureSize": 3, "L2gradient": True},  #
    {"tresh1": 100, "tresh2": 200, "apertureSize": 3, "L2gradient": False},  #
    {"tresh1": 100, "tresh2": 200, "apertureSize": 5, "L2gradient": True},  #
    {"tresh1": 100, "tresh2": 200, "apertureSize": 5, "L2gradient": False},  #
]

# =========================================================================================================== MAIN CODE

# Step 1: Reading RAW Image and converting to BGR for OpenCV (Read RAW Image) ===============================
# # Get Processed Image
rawSet = f.Converter.raw2bgr(pathRawImage)


# Step 2: Pre-process image, create "Ground Truth Image" (Grayscale, Resize, Save) ==========================
raw2GraySet = f.Converter.bgr2gray(rawSet)  # Convert to gray
scaledRaw2GraySet = f.Image.scaleSetBy(raw2GraySet, SCALE_FACTOR)  # Scale to new size
scaledRawSet = f.Image.scaleSetBy(rawSet, SCALE_FACTOR)  # Scale to new size RGB

# Save the processed image object to tagged image file TIFF as ground truth image
if SAVE_NEW_IMAGE:
    f.Image.saveSet(
        scaledRaw2GraySet, pathGTIGrayScaled
    )  # -------------------------------------------------------------------------------------------------------------------------- SAVE
    f.Image.saveSet(
        scaledRawSet, pathGTIRGBScaled
    )  # -------------------------------------------------------------------------------------------------------------------------- SAVE

# Read the ground truth image
gtiGraySet = f.Image.readSet(pathGTIGrayScaled)
gtiRGBSet = f.Image.readSet(pathGTIRGBScaled)

# Step 3: Apply blur Smooth Filter ========================================================================
sfGraySet = f.SmoothFilter.applyFilter(
    gtiGraySet,
    SMOOTH_TYPE,
    SMOOTH_KERNEL_SIZE_X,
    SMOOTH_KERNEL_SIZE_Y,
    SMOOTH_KERNEL_SIZE,
    SMOOTH_SIGMA_X,
    SMOOTH_SIGMA_Y,
    SMOOTH_D,
    SMOOTH_SIGMA_COLOR,
    SMOOTH_SIGMA_SPACE,
    SMOOTH_H,
    SMOOTH_SEARCH_WINDOW_SIZE,
)
sfRGBSet = f.SmoothFilter.applyFilter(
    gtiRGBSet,
    SMOOTH_TYPE,
    SMOOTH_KERNEL_SIZE_X,
    SMOOTH_KERNEL_SIZE_Y,
    SMOOTH_KERNEL_SIZE,
    SMOOTH_SIGMA_X,
    SMOOTH_SIGMA_Y,
    SMOOTH_D,
    SMOOTH_SIGMA_COLOR,
    SMOOTH_SIGMA_SPACE,
    SMOOTH_H,
    SMOOTH_SEARCH_WINDOW_SIZE,
)
# sfGraySet = f.Converter.gray2bgr(sfGraySet)
# sfGraySet = f.Converter.gray2rgb(sfGraySet)
# sfGraySet = f.Converter.bgr2gray(sfGraySet)
# sfGraySet = f.Converter.gray2bgr(sfGraySet)

# Step 4: Apply Edge Filter ===============================================================================
# 1. Canny edge detection | sf1CannySet[i][j] = i-th image, j-th Canny parameter
# A. Without smoothing
cannySFGraySet = f.EdgeFilter.applyCanny(sfGraySet, cannyParamSet)
cannySFRGBSet = f.EdgeFilter.applyCanny(sfRGBSet, cannyParamSet)
cannyGraySet = f.EdgeFilter.applyCanny(gtiGraySet, cannyParamSet)
cannyRGBSet = f.EdgeFilter.applyCanny(gtiRGBSet, cannyParamSet)

# Step 5: Compute Peak Signal-to-Noise Ratio (PSNR), compression ratio ======================================
# psnr = [
#     Compute.peakSignalToNoiseRatio(groundTruthImageSet, imgSet, 2)
#     for imgSet in sfFullSet
# ]

# cr = [Compute.compressionRatio(groundTruthImageSet, imgSet, 2) for imgSet in sfFullSet]

# Step 6: Output the results ===============================================================================+
set1 = cannyGraySet[IMG_NO]
set2 = cannySFGraySet[IMG_NO]
set3 = cannyRGBSet[IMG_NO]
set4 = cannySFRGBSet[IMG_NO]

# Add reference image to the set
set1.insert(0, gtiGraySet[IMG_NO])
set2.insert(0, sfGraySet[IMG_NO])
set3.insert(0, gtiRGBSet[IMG_NO])
set4.insert(0, sfRGBSet[IMG_NO])

set1 = f.Image.concatSet(set1, axis=1)
set2 = f.Image.concatSet(set2, axis=1)
set3 = f.Image.concatSet(set3, axis=1)
set4 = f.Image.concatSet(set4, axis=1)

sideBySideA = f.Image.concat(set1, set3, axis=0)
sideBySideB = f.Image.concat(set2, set4, axis=0)
sideBySide = f.Image.concat(sideBySideA, sideBySideB, axis=0)

f.Image.showSet(sideBySide, "Filter")


cv2.waitKey(0)
cv2.destroyAllWindows()
