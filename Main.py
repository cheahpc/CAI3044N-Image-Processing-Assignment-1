import func as f
import cv2

# Set resize scale factor
newWidth = 1000
newHeight = 1000
scaleFactor = 0.0908
filterKernelSize = 5  # Value must be odd number and greater than 1
img = 2
saveNewImage = False

# Set Image path in array
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

# Step 1: Reading RAW Image and converting to BGR for OpenCV (Read RAW Image) ===============================
# # Get Processed Image
rawSet = f.Converter.raw2bgr(pathRawImage)
rawScaledImageSet = f.Image.scaleSetBy(rawSet, scaleFactor)

# Step 2: Pre-process image, create "Ground Truth Image" (Grayscale, Resize, Save) ==========================
raw2GraySet = f.Converter.bgr2gray(rawSet)
scaledRaw2GraySet = f.Image.scaleSetBy(raw2GraySet, scaleFactor)
scaledRawSet = f.Image.scaleSetBy(rawSet, scaleFactor)

# Save the processed image object to tagged image file TIFF as ground truth image
if saveNewImage:
    f.Image.saveSet(
        scaledRaw2GraySet, pathGTIGrayScaled
    )  # -------------------------------------------------------------------------------------------------------------------------- SAVE
    f.Image.saveSet(
        scaledRawSet, pathGTIRGBScaled
    )  # -------------------------------------------------------------------------------------------------------------------------- SAVE

# Read the saved ground truth image
gtiGraySet = f.Image.readSet(pathGTIGrayScaled)
gtiRGBSet = f.Image.readSet(pathGTIRGBScaled)

# Prepare the image
gtiGraySet = f.Converter.rgb2gray(gtiGraySet)

# Step 3: Apply blur filter to the image (Smoothing Filter) =================================================
# 1. Average filter
sfAverageGraySet = f.SmoothFilter.applyAverageFilter(
    gtiGraySet, filterKernelSize, filterKernelSize
)
sfAverageRGBSet = f.SmoothFilter.applyAverageFilter(
    gtiRGBSet, filterKernelSize, filterKernelSize
)

# 2. Box filter
sfBoxGraySet = f.SmoothFilter.applyBoxFilter(
    gtiGraySet, filterKernelSize, filterKernelSize
)
sfBoxRGBSet = f.SmoothFilter.applyBoxFilter(
    gtiRGBSet, filterKernelSize, filterKernelSize
)

# 3. Gaussian filter
sfGaussianGraySet = f.SmoothFilter.applyGaussianBlurFilter(
    gtiGraySet, filterKernelSize, filterKernelSize
)
sfGaussianRGBSet = f.SmoothFilter.applyGaussianBlurFilter(
    gtiRGBSet, filterKernelSize, filterKernelSize
)

# 4. Median filter
sfMedianGraySet = f.SmoothFilter.applyMedianBlurFilter(gtiGraySet, filterKernelSize)
sfMedianRGBSet = f.SmoothFilter.applyMedianBlurFilter(gtiRGBSet, filterKernelSize)

# 5. Bilateral filter
sfBilateralGraySet = f.SmoothFilter.applyBilateralFilter(gtiGraySet, 9, 75, 75)
sfBilateralRGBSet = f.SmoothFilter.applyBilateralFilter(gtiRGBSet, 9, 75, 75)

# 6. Non-local means filter
sfNlMeanGraySet = f.SmoothFilter.applyNonLocalMeansFilter(gtiGraySet, 10, 20)
sfNlMeanRGBSet = f.SmoothFilter.applyNonLocalMeansFilter(gtiRGBSet, 10, 20)

# 7. Custom kernel filter
sfCustKernelGraySet = f.SmoothFilter.applyCustomKernelFilter(
    gtiGraySet, filterKernelSize
)
sfCustKernelRGBSet = f.SmoothFilter.applyCustomKernelFilter(gtiRGBSet, filterKernelSize)

# Compile all filtered set
sfFullGraySet = [
    sfAverageGraySet,  # sf1
    sfBoxGraySet,  # sf2
    sfGaussianGraySet,  # sf3
    sfMedianGraySet,  # sf4
    sfBilateralGraySet,  # sf5
    sfNlMeanGraySet,  # sf6
    sfCustKernelGraySet,  # sf7
]
sfFullRGBSet = [
    sfAverageRGBSet,  # sf1
    sfBoxRGBSet,  # sf2
    sfGaussianRGBSet,  # sf3
    sfMedianRGBSet,  # sf4
    sfBilateralRGBSet,  # sf5
    sfNlMeanRGBSet,  # sf6
    sfCustKernelRGBSet,  # sf7
]

# Step 4: Apply edge detection filter to the image (Edge Detection Filter) ==================================
# 1. Canny edge detection | sf1CannySet[i][j] = i-th image, j-th Canny parameter
# A. Without smoothing
cannyGraySet = f.EdgeFilter.applyCanny(gtiGraySet, cannyParamSet)  # Without smoothing
cannyRGBSet = f.EdgeFilter.applyCanny(gtiRGBSet, cannyParamSet)  # Without smoothing

# B. Average filter
sf1CannyGraySet = f.EdgeFilter.applyCanny(sfAverageGraySet, cannyParamSet)
sf1CannyRGBSet = f.EdgeFilter.applyCanny(sfAverageRGBSet, cannyParamSet)

# C. Box filter
sf2CannyGraySet = f.EdgeFilter.applyCanny(sfBoxGraySet, cannyParamSet)
sf2CannyRGBSet = f.EdgeFilter.applyCanny(sfBoxRGBSet, cannyParamSet)

# D. Gaussian filter
sf3CannyGraySet = f.EdgeFilter.applyCanny(sfGaussianGraySet, cannyParamSet)
sf3CannyRGBSet = f.EdgeFilter.applyCanny(sfGaussianRGBSet, cannyParamSet)

# E. Median filter
sf4CannyGraySet = f.EdgeFilter.applyCanny(sfMedianGraySet, cannyParamSet)
sf4CannyRGBSet = f.EdgeFilter.applyCanny(sfMedianRGBSet, cannyParamSet)

# F. Bilateral filter
sf5CannyGraySet = f.EdgeFilter.applyCanny(sfBilateralGraySet, cannyParamSet)
sf5CannyRGBSet = f.EdgeFilter.applyCanny(sfBilateralRGBSet, cannyParamSet)

# G. Non-local means filter
sf6CannyGraySet = f.EdgeFilter.applyCanny(sfNlMeanGraySet, cannyParamSet)
sf6CannyRGBSet = f.EdgeFilter.applyCanny(sfNlMeanRGBSet, cannyParamSet)

# H. Custom kernel filter
sf7CannyGraySet = f.EdgeFilter.applyCanny(sfCustKernelGraySet, cannyParamSet)
sf7CannyRGBSet = f.EdgeFilter.applyCanny(sfCustKernelRGBSet, cannyParamSet)

# Step 5: Compute Peak Signal-to-Noise Ratio (PSNR), compression ratio =======================================

# psnr = [
#     Compute.peakSignalToNoiseRatio(groundTruthImageSet, imgSet, 2)
#     for imgSet in sfFullSet
# ]

# cr = [Compute.compressionRatio(groundTruthImageSet, imgSet, 2) for imgSet in sfFullSet]

# Declare image set, gray and rgb
set1 = cannyGraySet[img]
set2 = cannyRGBSet[img]
set3 = sf1CannyGraySet[img]
set4 = sf1CannyRGBSet[img]

# Convert gray channel to 3 channel
set2 = f.Converter.gray2bgr(set2)
set4 = f.Converter.gray2bgr(set4)

# Add reference image to the set
set1.insert(0, gtiGraySet[img])
set2.insert(0, gtiRGBSet[img])
set3.insert(0, sfAverageGraySet[img])
set4.insert(0, sfAverageRGBSet[img])

# Convert all to 3 channel
set1 = f.Converter.gray2rgb(set1)
set3 = f.Converter.gray2rgb(set3)

set1 = f.Image.concatSet(set1, axis=1)
set2 = f.Image.concatSet(set2, axis=1)
set3 = f.Image.concatSet(set3, axis=1)
set4 = f.Image.concatSet(set4, axis=1)

sideBySideA = f.Image.concat(set1, set3, axis=0)
sideBySideB = f.Image.concat(set2, set4, axis=0)
sideBySide = f.Image.concat(sideBySideA, sideBySideB, axis=0)

f.Image.showSet(sideBySide, "Filter")

# Smooth Filtered
# Image.show(smoothFilteredSet[0], "Smooth Filtered Image")  # Average Filtered Image
# Image.show(smoothFilteredSet[1], "Smooth Filtered Image") # Box Filtered Image
# Image.show(smoothFilteredSet[2], "Smooth Filtered Image") # Gaussian Filtered Image
# Image.show(smoothFilteredSet[3], "Smooth Filtered Image") # Median Filtered Image
# Image.show(smoothFilteredSet[4], "Smooth Filtered Image") # Bilateral Filtered Image
# Image.show(smoothFilteredSet[5], "Smooth Filtered Image") # Non-Local Means Filtered Image
# Image.show(smoothFilteredSet[6], "Smooth Filtered Image") # Custom Kernel Filtered Image

# Edge Detected Filtered
# Image.show(edgeDetectedSet, "Edge Detected Filtered Image")

# Concatenated
# Image.show(gtiXfilteredSet, "Average Filtered Image")

cv2.waitKey(0)
cv2.destroyAllWindows()
