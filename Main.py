import func as f
import cv2

# Settings
SAVE_NEW_GTI = False
SAVE_NEW_PROCESSED = True
SCALE_FACTOR = 0.25


path_raw_h = "img/raw_h_1.NEF"
path_raw_v = "img/raw_v_2.NEF"

path_gti_h = "img/gti_h.TIFF"  # Gray Scaled
path_gti_v = "img/gti_v.TIFF"  # Gray Scaled


# =========================================================================================================== MAIN CODE

# Step 1: Read RAW Image ====================================================================================
raw_h = f.Converter.raw2bgr(path_raw_h)  # 3 channel
raw_v = f.Converter.raw2bgr(path_raw_v)  # 3 channel

# Step 2: Convert to Gray ===================================================================================
raw_h_gray = f.Converter.bgr2gray(raw_h)  # Convert to gray, 1 channel
raw_v_gray = f.Converter.bgr2gray(raw_v)  # Convert to gray, 1 channel

# Step 3: Scale the image ===================================================================================
raw_h_gray_scaled = f.Image.scaleSetBy(raw_h_gray, SCALE_FACTOR)  # 1 channel, scaled
raw_v_gray_scaled = f.Image.scaleSetBy(raw_v_gray, SCALE_FACTOR)  # 1 channel, scaled

# Save as TIFF for GTI (Ground Truth Image)
# -------------------------------------------------------------------------------------------------------------------------- SAVE
if SAVE_NEW_GTI:
    f.Image.saveSet(raw_h_gray_scaled, path_gti_h)
    f.Image.saveSet(raw_v_gray_scaled, path_gti_v)

# Step 4: Read GTI Image ====================================================================================
gti_h = f.Image.readSet(path_gti_h, True)  # Read image as gray scaled
gti_v = f.Image.readSet(path_gti_v, True)  # Read image as gray scaled


# Step 5: Apply Smooth Filter ===============================================================================
# Apply Gaussian
sg_h_1 = f.SmoothFilter.applyGaussianBlurFilter(gti_h, 3, 0, 0)
sg_h_2 = f.SmoothFilter.applyGaussianBlurFilter(gti_h, 5, 0, 0)
sg_h_3 = f.SmoothFilter.applyGaussianBlurFilter(gti_h, 7, 0, 0)
sg_h_11 = f.SmoothFilter.applyGaussianBlurFilter(gti_h, 3, 5, 5)
sg_h_12 = f.SmoothFilter.applyGaussianBlurFilter(gti_h, 3, 20, 20)
sg_h_13 = f.SmoothFilter.applyGaussianBlurFilter(gti_h, 3, 50, 50)

sg_v_1 = f.SmoothFilter.applyGaussianBlurFilter(gti_v, 3, 0, 0)
sg_v_2 = f.SmoothFilter.applyGaussianBlurFilter(gti_v, 5, 0, 0)
sg_v_3 = f.SmoothFilter.applyGaussianBlurFilter(gti_v, 7, 0, 0)
sg_v_11 = f.SmoothFilter.applyGaussianBlurFilter(gti_v, 3, 5, 5)
sg_v_12 = f.SmoothFilter.applyGaussianBlurFilter(gti_v, 3, 20, 20)
sg_v_13 = f.SmoothFilter.applyGaussianBlurFilter(gti_v, 3, 50, 50)

# Apply Bilateral


# Step 6: Apply Edge Filter ===============================================================================
# canny1 = f.EdgeFilter.applyCanny(gti_h, cannyParamSet[0])
# canny2 = f.EdgeFilter.applyCanny(gti_v, cannyParamSet[0])

# Step 5: Compute Peak Signal-to-Noise Ratio (PSNR) and compression ratio ===================================
# PSNR - Smoothed
psnr_sg_h_1 = f.Compute.peakSignalToNoiseRatio(gti_h, sg_h_1)
psnr_sg_h_2 = f.Compute.peakSignalToNoiseRatio(gti_h, sg_h_2)
psnr_sg_h_3 = f.Compute.peakSignalToNoiseRatio(gti_h, sg_h_3)

psnr_sg_v_1 = f.Compute.peakSignalToNoiseRatio(gti_v, sg_v_1)
psnr_sg_v_2 = f.Compute.peakSignalToNoiseRatio(gti_v, sg_v_2)
psnr_sg_v_3 = f.Compute.peakSignalToNoiseRatio(gti_v, sg_v_3)


# Compression Ratio - RAW & GTI
cr_raw_gti_h = f.Compute.compressionRatio(raw_h, gti_h, 8)
cr_raw_gti_v = f.Compute.compressionRatio(raw_v, gti_v, 8)


# Step 6: Output the results ===============================================================================+

# ----------------------------------------------------------------------------------------------------------- Compilation

s_g_h_list = [
    gti_h,  # Original
    # sg_h_1,
    # sg_h_2,
    # sg_h_3,
    sg_h_11,
    sg_h_12,
    sg_h_13,
]

s_g_v_list = [
    gti_v,  # Original
    # sg_v_1,
    # sg_v_2,
    # sg_v_3,
    sg_v_11,
    sg_v_12,
    sg_v_13,
]

s_g_h = f.Image.concatSet(s_g_h_list, axis=0)
s_g_v = f.Image.concatSet(s_g_v_list, axis=1)

if SAVE_NEW_PROCESSED:
    f.Image.saveSet(s_g_h, "img/s_g_h.TIFF")
    f.Image.saveSet(s_g_v, "img/s_g_v.TIFF")

# ----------------------------------------------------------------------------------------------------------- Show - Information
# Information
print("---------- Informations ----------")
print("raw_h image shape: ", raw_h.shape, " | Type: ", f.Image.getImageType(raw_h), " | Size: ", raw_h.size)
print("raw_v image shape: ", raw_v.shape, " | Type: ", f.Image.getImageType(raw_v), " | Size: ", raw_v.size)
print("gti_h image shape: ", gti_h.shape, " | Type: ", f.Image.getImageType(gti_h), " | Size: ", gti_h.size)
print("gti_v image shape: ", gti_v.shape, " | Type: ", f.Image.getImageType(gti_v), " | Size: ", gti_v.size)
print()
# ----------------------------------------------------------------------------------------------------------- Show - PSNR
print("---------- PSNR ----------")
# Peak Signal To Noise Ratio - Original
# Peak Signal To Noise Ratio - Smoothed
print("PSNR GTI X Smoothed Gaussian 1 (Horizontal):", psnr_sg_h_1)
print("PSNR GTI X Smoothed Gaussian 2 (Horizontal):", psnr_sg_h_2)
print("PSNR GTI X Smoothed Gaussian 3 (Horizontal):", psnr_sg_h_3)
print()

# ----------------------------------------------------------------------------------------------------------- Show Compression Ratio
print("---------- Compression Ratio ----------")
# Compression Ratio - Scaled
print("Compression Ratio RAW X GTI (Horizontal): ", cr_raw_gti_h)
print("Compression Ratio Raw X GTI (Vertical): ", cr_raw_gti_v)
print()

# ----------------------------------------------------------------------------------------------------------- Show Image
# Show Image - RAW
# cv2.imshow("raw_h", raw_h)
# cv2.imshow("raw_v", raw_v)

# Show Image - GTI
# cv2.imshow("gti_h", gti_h)
# cv2.imshow("gti_v", gti_v)

# Show Image - Smoothed - Gaussian
# cv2.imshow("s_g_h", s_g_h)
# cv2.imshow("s_g_v", s_g_v)


cv2.waitKey(0)
cv2.destroyAllWindows()

exit()
# ----------------------------------------------------------------------------------------------------------- Show
