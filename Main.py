import func as f
import cv2

# Settings
IMG_NO = 2
READ_SET = False
SAVE_NEW_IMAGE = False

SCALE_FACTOR = 0.15

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
    "img/gti_gray_v_1.TIFF",
    "img/gti_gray_v_2.TIFF",
    "img/gti_gray_v_3.TIFF",
    "img/gti_gray_v_4.TIFF",
    "img/gti_gray_h_1.TIFF",
    "img/gti_gray_h_2.TIFF",
    "img/gti_gray_h_3.TIFF",
    "img/gti_gray_h_4.TIFF",
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

# Canny parameters
cannyParamSet = [
    {"tresh1": 100, "tresh2": 200, "apertureSize": 3, "L2gradient": True},  #
    {"tresh1": 100, "tresh2": 200, "apertureSize": 3, "L2gradient": False},  #
    {"tresh1": 100, "tresh2": 200, "apertureSize": 5, "L2gradient": True},  #
    {"tresh1": 100, "tresh2": 200, "apertureSize": 5, "L2gradient": False},  #
]

# Laplacian parameters
laplacianParamSet = [
    {
        "ddepth": cv2.CV_16S,
        "ksize": 3,
        "scale": 1,
        "delta": 0,
        "borderType": cv2.BORDER_DEFAULT,
    },  #
    {
        "ddepth": cv2.CV_16S,
        "ksize": 3,
        "scale": 1,
        "delta": 0,
        "borderType": cv2.BORDER_DEFAULT,
    },  #
]

# =========================================================================================================== MAIN CODE

# Step 1: Read RAW Image ====================================================================================
raw = [f.Converter.raw2bgr(pathRawImage[IMG_NO]), f.Converter.raw2bgr(pathRawImage)][READ_SET]  # 3 channel image, 3 dimension (Height, Width, Channel)

# Step 2: Convert to Gray ===================================================================================
rawGray = f.Converter.bgr2gray(raw)  # Convert to gray, 1 channel image, 2 dimension (Height, Width)

# Step 3: Scale the image ===================================================================================
rawGrayScaled = f.Image.scaleSetBy(rawGray, SCALE_FACTOR)  # 1 channel image, 2 dimension (Height, Width)
rawScaled = f.Image.scaleSetBy(raw, SCALE_FACTOR)  # Scale to new size RGB

# Save as TIFF for GTI (Ground Truth Image)
# -------------------------------------------------------------------------------------------------------------------------- SAVE
if SAVE_NEW_IMAGE:
    f.Image.saveSet(rawGrayScaled, pathGTIGrayScaled, True)
    f.Image.saveSet(rawScaled, pathGTIRGBScaled)

# Step 4: Read GTI Image ====================================================================================
gtiGray = [f.Image.readSet(pathGTIGrayScaled[IMG_NO], True), f.Image.readSet(pathGTIGrayScaled, True)][READ_SET]
gtiRGB = [f.Image.readSet(pathGTIRGBScaled[IMG_NO]), f.Image.readSet(pathGTIRGBScaled)][READ_SET]

# Step 5: Apply blur Smooth Filter ========================================================================
sfAverageRGB = f.SmoothFilter.applyAverageFilter(gtiRGB[IMG_NO], SMOOTH_KERNEL_SIZE_X, SMOOTH_KERNEL_SIZE_Y)
sfGaussianRGB1 = f.SmoothFilter.applyGaussianBlurFilter(gtiRGB[IMG_NO], 3, 3, 0, 0)
sfGaussianRGB2 = f.SmoothFilter.applyGaussianBlurFilter(gtiRGB[IMG_NO], 11, 11, 0, 0)
sfGaussianRGB3 = f.SmoothFilter.applyGaussianBlurFilter(gtiRGB[IMG_NO], 3, 3, 25, 25)
sfBilateralRGB1 = f.SmoothFilter.applyBilateralFilter(gtiRGB[IMG_NO], 9, 100, 100)
sfBilateralRGB2 = f.SmoothFilter.applyBilateralFilter(gtiRGB[IMG_NO], 11, 75, 75)

# ----------------------------------------------------------------------------------------------------------- Show
print("gtiGray image: ", gtiGray.shape, " | ", f.Image.getImageType(gtiGray))
print("gtiRGB image: ", gtiRGB.shape, " | ", f.Image.getImageType(gtiRGB))

# Show Image
cv2.imshow("gtiGray", gtiGray)
cv2.imshow("gtiRGB", gtiRGB)
cv2.waitKey(0)
cv2.destroyAllWindows()

exit()
# ----------------------------------------------------------------------------------------------------------- Show

# Step 6: Apply Edge Filter ===============================================================================
canny1 = f.EdgeFilter.applyCanny(sfBilateralRGB1, cannyParamSet[0])
canny2 = f.EdgeFilter.applyCanny(sfBilateralRGB2, cannyParamSet[0])

# Step 5: Compute Peak Signal-to-Noise Ratio (PSNR), compression ratio ======================================
# psnr = [
#     Compute.peakSignalToNoiseRatio(groundTruthImageSet, imgSet, 2)
#     for imgSet in sfFullSet
# ]

# cr = [Compute.compressionRatio(groundTruthImageSet, imgSet, 2) for imgSet in sfFullSet]

# Step 6: Output the results ===============================================================================+

# For Smooth Filter ---------------------------------------------------------------------
# set1 = [sfBilateralRGB1, canny1, sfBilateralRGB2, canny2]
# set1 = cannyGraySet[IMG_NO]
# set2 = cannySFGraySet[IMG_NO]
# set3 = cannyRGBSet[IMG_NO]
# set4 = cannySFRGBSet[IMG_NO]

# Add reference image to the set
# set1.insert(0, gtiGraySet[IMG_NO])
# set2.insert(0, sfGraySet[IMG_NO])
# set3.insert(0, gtiRGBSet[IMG_NO])
# set4.insert(0, sfRGBSet[IMG_NO])

# set1 = f.Image.concatSet(set1, axis=1)
# set2 = f.Image.concatSet(set2, axis=1)
# set3 = f.Image.concatSet(set3, axis=1)
# set4 = f.Image.concatSet(set4, axis=1)

# sideBySideA = f.Image.concat(set1, set3, axis=0)
# sideBySideB = f.Image.concat(set2, set4, axis=0)
# sideBySide = f.Image.concat(sideBySideA, sideBySideB, axis=0)

# f.Image.showSet(set1, "Smooth Filter")


cv2.waitKey(0)
cv2.destroyAllWindows()
