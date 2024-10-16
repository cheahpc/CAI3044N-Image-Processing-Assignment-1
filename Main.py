import rawpy
import numpy as np
import cv2

# Settings
SAVE_NEW_GTI = False
SAVE_NEW_SMOOTHED = True
SAVE_NEW_CANNY = False
SAVE_NEW_LAPLACIAN = False
SCALE_FACTOR = 0.25

path_raw_h = "img/raw_h_1.NEF"
path_raw_v = "img/raw_v_2.NEF"

path_gti_h = "img/gti_h.TIFF"  # Gray Scaled
path_gti_v = "img/gti_v.TIFF"  # Gray Scaled

path_sg_h = "img/sg_h.TIFF"  # Gray Scaled
path_sg_v = "img/sg_v.TIFF"  # Gray Scaled
path_sb_h = "img/sb_h.TIFF"  # Gray Scaled
path_sb_v = "img/sb_v.TIFF"  # Gray Scaled

# =========================================================================================================== MAIN CODE

# Step 1: Read RAW Image ====================================================================================
raw_h = cv2.cvtColor(rawpy.imread(path_raw_h).postprocess(use_camera_wb=True), cv2.COLOR_RGB2BGR)  # 3 channel
raw_v = cv2.cvtColor(rawpy.imread(path_raw_v).postprocess(use_camera_wb=True), cv2.COLOR_RGB2BGR)  # 3 channel

# Step 2: Convert to Gray ===================================================================================
raw_h_gray = cv2.cvtColor(raw_h, cv2.COLOR_BGR2GRAY)  # Convert to gray, 1 channel
raw_v_gray = cv2.cvtColor(raw_v, cv2.COLOR_BGR2GRAY)  # Convert to gray, 1 channel

# Step 3: Scale the image ===================================================================================
raw_h_gray_scaled = cv2.resize(raw_h_gray, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)  # 1 channel, scaled
raw_v_gray_scaled = cv2.resize(raw_v_gray, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)  # 1 channel, scaled

# Save as TIFF for GTI (Ground Truth Image)
# -------------------------------------------------------------------------------------------------------------------------- SAVE
if SAVE_NEW_GTI:
    cv2.imwrite(path_gti_h, raw_h_gray_scaled)
    cv2.imwrite(path_gti_v, raw_v_gray_scaled)

# Step 4: Read GTI Image ====================================================================================
gti_h = cv2.imread(path_gti_h, 0)  # Read image as gray scaled
gti_v = cv2.imread(path_gti_v, 0)  # Read image as gray scaled


# Step 5: Apply Smooth Filter ===============================================================================
# Apply Gaussian
sg_h_1 = cv2.GaussianBlur(gti_h, (3, 3), 0, 0)
sg_h_2 = cv2.GaussianBlur(gti_h, (5, 5), 0, 0)
sg_h_3 = cv2.GaussianBlur(gti_h, (7, 7), 0, 0)
sg_v_1 = cv2.GaussianBlur(gti_v, (3, 3), 0, 0)
sg_v_2 = cv2.GaussianBlur(gti_v, (5, 5), 0, 0)
sg_v_3 = cv2.GaussianBlur(gti_v, (7, 7), 0, 0)


# Apply Bilateral
sb_h_1 = cv2.bilateralFilter(gti_h, 3, 75, 75)
sb_h_2 = cv2.bilateralFilter(gti_h, 5, 75, 75)
sb_h_3 = cv2.bilateralFilter(gti_h, 7, 75, 75)
sb_v_1 = cv2.bilateralFilter(gti_v, 3, 75, 75)
sb_v_2 = cv2.bilateralFilter(gti_v, 5, 75, 75)
sb_v_3 = cv2.bilateralFilter(gti_v, 7, 75, 75)

# Step 6: Apply Edge Filter ===============================================================================
# Apply Canny
# ec_h_1 = cv2.Canny(gti_h,100,200,3, L2gradient=True)
# canny2 = f.EdgeFilter.applyCanny(gti_v, cannyParamSet[0])

# Step 5: Compute Peak Signal-to-Noise Ratio (PSNR) and compression ratio ===================================
# PSNR - Smoothed Gaussian
psnr_sg_h_1 = round(cv2.PSNR(gti_h, sg_h_1), 4)
psnr_sg_h_2 = round(cv2.PSNR(gti_h, sg_h_2), 4)
psnr_sg_h_3 = round(cv2.PSNR(gti_h, sg_h_3), 4)
psnr_sg_v_1 = round(cv2.PSNR(gti_v, sg_v_1), 4)
psnr_sg_v_2 = round(cv2.PSNR(gti_v, sg_v_2), 4)
psnr_sg_v_3 = round(cv2.PSNR(gti_v, sg_v_3), 4)


# PSNR - Smoothed Bilateral
psnr_sb_h_1 = round(cv2.PSNR(gti_h, sb_h_1), 4)
psnr_sb_h_2 = round(cv2.PSNR(gti_h, sb_h_2), 4)
psnr_sb_h_3 = round(cv2.PSNR(gti_h, sb_h_3), 4)
psnr_sb_v_1 = round(cv2.PSNR(gti_v, sb_v_1), 4)
psnr_sb_v_2 = round(cv2.PSNR(gti_v, sb_v_2), 4)
psnr_sb_v_3 = round(cv2.PSNR(gti_v, sb_v_3), 4)


# Compression Ratio - RAW & GTI
cr_raw_gti_h = round(raw_h.size / gti_h.size, 4)
cr_raw_gti_v = round(raw_v.size / gti_v.size, 4)


# Step 6: Output the results ===============================================================================+

# ----------------------------------------------------------------------------------------------------------- Compilation

sg_h_list = [
    gti_h,  # Original
    sg_h_1,
    sg_h_2,
    sg_h_3,
]

sg_v_list = [
    gti_v,  # Original
    sg_v_1,
    sg_v_2,
    sg_v_3,
]

sb_h_list = [
    gti_h,  # Original
    sb_h_1,
    sb_h_2,
    sb_h_3,
]

sb_v_list = [
    gti_v,  # Original
    sb_v_1,
    sb_v_2,
    sb_v_3,
]

sg_h = np.concatenate(sg_h_list, 1)
sg_v = np.concatenate(sg_v_list, 1)
sb_h = np.concatenate(sb_h_list, 1)
sb_v = np.concatenate(sb_v_list, 1)

if SAVE_NEW_SMOOTHED:
    cv2.imwrite(path_sg_h, sg_h)
    cv2.imwrite(path_sg_v, sg_v)
    cv2.imwrite(path_sb_h, sb_h)
    cv2.imwrite(path_sb_v, sb_v)


# ----------------------------------------------------------------------------------------------------------- Show - Information
# Information
print("---------- Informations ----------")
print("raw_h image shape: ", raw_h.shape, " | Type: ", raw_h.dtype, " | Size: ", raw_h.size)
print("raw_v image shape: ", raw_v.shape, " | Type: ", raw_v.dtype, " | Size: ", raw_v.size)
print("gti_h image shape: ", gti_h.shape, " | Type: ", gti_h.dtype, " | Size: ", gti_h.size)
print("gti_v image shape: ", gti_v.shape, " | Type: ", gti_v.dtype, " | Size: ", gti_v.size)
print()
# ----------------------------------------------------------------------------------------------------------- Show - PSNR
print("---------- PSNR ----------")
# Peak Signal To Noise Ratio - Smoothed
print("PSNR - GTI X Smoothed - [Gaussian Blur] 1 (Horizontal):", psnr_sg_h_1, " dB")
print("PSNR - GTI X Smoothed - [Gaussian Blur] 2 (Horizontal):", psnr_sg_h_2, " dB")
print("PSNR - GTI X Smoothed - [Gaussian Blur] 3 (Horizontal):", psnr_sg_h_3, " dB")
print("PSNR - GTI X Smoothed - [Gaussian Blur] 1 (Vertical):", psnr_sg_v_1, " dB")
print("PSNR - GTI X Smoothed - [Gaussian Blur] 2 (Vertical):", psnr_sg_v_2, " dB")
print("PSNR - GTI X Smoothed - [Gaussian Blur] 3 (Vertical):", psnr_sg_v_3, " dB")
print()

print("PSNR - GTI X Smoothed - [Bilateral Filtering] 1 (Horizontal):", psnr_sb_h_1, " dB")
print("PSNR - GTI X Smoothed - [Bilateral Filtering] 2 (Horizontal):", psnr_sb_h_2, " dB")
print("PSNR - GTI X Smoothed - [Bilateral Filtering] 3 (Horizontal):", psnr_sb_h_3, " dB")
print("PSNR - GTI X Smoothed - [Bilateral Filtering] 1 (Vertical):", psnr_sb_v_1, " dB")
print("PSNR - GTI X Smoothed - [Bilateral Filtering] 2 (Vertical):", psnr_sb_v_2, " dB")
print("PSNR - GTI X Smoothed - [Bilateral Filtering] 3 (Vertical):", psnr_sb_v_3, " dB")
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
