import rawpy
import numpy as np
import cv2
import os

# Settings
SAVE_NEW_GTI = True

SAVE_NEW_SMOOTHED = True

SAVE_NEW_CANNY = True
SAVE_NEW_LAPLACIAN = True

SCALE_FACTOR = 0.25

path_raw_h = "img/raw_h_1.NEF"
path_raw_v = "img/raw_v_2.NEF"

path_gti_h = "img/gti_h.TIFF"
path_gti_v = "img/gti_v.TIFF"

path_sg_h = "img/sg_h.TIFF"
path_sg_h_1 = "img/sg_h_1.TIFF"
path_sg_h_2 = "img/sg_h_2.TIFF"
path_sg_h_3 = "img/sg_h_3.TIFF"
path_sg_v = "img/sg_v.TIFF"
path_sg_v_1 = "img/sg_v_1.TIFF"
path_sg_v_2 = "img/sg_v_2.TIFF"
path_sg_v_3 = "img/sg_v_3.TIFF"

path_sb_h = "img/sb_h.TIFF"
path_sb_h_1 = "img/sb_h_1.TIFF"
path_sb_h_2 = "img/sb_h_2.TIFF"
path_sb_h_3 = "img/sb_h_3.TIFF"
path_sb_v = "img/sb_v.TIFF"
path_sb_v_1 = "img/sb_v_1.TIFF"
path_sb_v_2 = "img/sb_v_2.TIFF"
path_sb_v_3 = "img/sb_v_3.TIFF"

# Edge Detection
path_ec_h = "img/ec_h.TIFF"
path_ec_h_1 = "img/ec_h_1.TIFF"
path_ec_h_2 = "img/ec_h_2.TIFF"
path_ec_h_3 = "img/ec_h_3.TIFF"
path_ec_h_4 = "img/ec_h_4.TIFF"
path_ec_h_5 = "img/ec_h_5.TIFF"
path_ec_h_6 = "img/ec_h_6.TIFF"
path_ec_h_7 = "img/ec_h_7.TIFF"
path_ec_h_8 = "img/ec_h_8.TIFF"
path_ec_v = "img/ec_v.TIFF"
path_ec_v_1 = "img/ec_v_1.TIFF"
path_ec_v_2 = "img/ec_v_2.TIFF"
path_ec_v_3 = "img/ec_v_3.TIFF"
path_ec_v_4 = "img/ec_v_4.TIFF"
path_ec_v_5 = "img/ec_v_5.TIFF"
path_ec_v_6 = "img/ec_v_6.TIFF"
path_ec_v_7 = "img/ec_v_7.TIFF"
path_ec_v_8 = "img/ec_v_8.TIFF"

path_el_h = "img/el_h.TIFF"
path_el_h_1 = "img/el_h_1.TIFF"
path_el_h_2 = "img/el_h_2.TIFF"
path_el_h_3 = "img/el_h_3.TIFF"
path_el_h_4 = "img/el_h_4.TIFF"
path_el_h_5 = "img/el_h_5.TIFF"
path_el_h_6 = "img/el_h_6.TIFF"
path_el_h_7 = "img/el_h_7.TIFF"
path_el_h_8 = "img/el_h_8.TIFF"
path_el_v = "img/el_v.TIFF"
path_el_v_1 = "img/el_v_1.TIFF"
path_el_v_2 = "img/el_v_2.TIFF"
path_el_v_3 = "img/el_v_3.TIFF"
path_el_v_4 = "img/el_v_4.TIFF"
path_el_v_5 = "img/el_v_5.TIFF"
path_el_v_6 = "img/el_v_6.TIFF"
path_el_v_7 = "img/el_v_7.TIFF"
path_el_v_8 = "img/el_v_8.TIFF"

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
ec_h_1 = cv2.Canny(sb_h_3, 25, 225)
ec_h_2 = cv2.Canny(sb_h_3, 50, 200)
ec_h_3 = cv2.Canny(sb_h_3, 75, 175)
ec_h_4 = cv2.Canny(sb_h_3, 100, 150)
ec_h_5 = cv2.Canny(sb_h_3, 25, 150)
ec_h_6 = cv2.Canny(sb_h_3, 50, 175)
ec_h_7 = cv2.Canny(sb_h_3, 75, 200)
ec_h_8 = cv2.Canny(sb_h_3, 100, 225)

ec_v_1 = cv2.Canny(sb_v_2, 25, 225)
ec_v_2 = cv2.Canny(sb_v_2, 50, 200)
ec_v_3 = cv2.Canny(sb_v_2, 75, 175)
ec_v_4 = cv2.Canny(sb_v_2, 100, 150)
ec_v_5 = cv2.Canny(sb_v_2, 25, 150)
ec_v_6 = cv2.Canny(sb_v_2, 50, 175)
ec_v_7 = cv2.Canny(sb_v_2, 75, 200)
ec_v_8 = cv2.Canny(sb_v_2, 100, 225)

# Apply Laplacian
el_h_1 = cv2.convertScaleAbs(cv2.Laplacian(sb_h_3, cv2.CV_16S, ksize=1, scale=1, delta=0))
el_h_2 = cv2.convertScaleAbs(cv2.Laplacian(sb_h_3, cv2.CV_16S, ksize=3, scale=1, delta=0))
el_h_3 = cv2.convertScaleAbs(cv2.Laplacian(sb_h_3, cv2.CV_16S, ksize=5, scale=1, delta=0))
el_h_4 = cv2.convertScaleAbs(cv2.Laplacian(sb_h_3, cv2.CV_16S, ksize=7, scale=1, delta=0))
el_h_5 = cv2.convertScaleAbs(cv2.Laplacian(sb_h_3, cv2.CV_16S, ksize=1, scale=2, delta=0))
el_h_6 = cv2.convertScaleAbs(cv2.Laplacian(sb_h_3, cv2.CV_16S, ksize=1, scale=8, delta=0))
el_h_7 = cv2.convertScaleAbs(cv2.Laplacian(sb_h_3, cv2.CV_16S, ksize=3, scale=2, delta=0))
el_h_8 = cv2.convertScaleAbs(cv2.Laplacian(sb_h_3, cv2.CV_16S, ksize=3, scale=8, delta=0))

el_v_1 = cv2.convertScaleAbs(cv2.Laplacian(sb_v_2, cv2.CV_16S, ksize=1, scale=1, delta=0))
el_v_2 = cv2.convertScaleAbs(cv2.Laplacian(sb_v_2, cv2.CV_16S, ksize=3, scale=1, delta=0))
el_v_3 = cv2.convertScaleAbs(cv2.Laplacian(sb_v_2, cv2.CV_16S, ksize=5, scale=1, delta=0))
el_v_4 = cv2.convertScaleAbs(cv2.Laplacian(sb_v_2, cv2.CV_16S, ksize=7, scale=1, delta=0))
el_v_5 = cv2.convertScaleAbs(cv2.Laplacian(sb_v_2, cv2.CV_16S, ksize=1, scale=2, delta=0))
el_v_6 = cv2.convertScaleAbs(cv2.Laplacian(sb_v_2, cv2.CV_16S, ksize=1, scale=8, delta=0))
el_v_7 = cv2.convertScaleAbs(cv2.Laplacian(sb_v_2, cv2.CV_16S, ksize=3, scale=2, delta=0))
el_v_8 = cv2.convertScaleAbs(cv2.Laplacian(sb_v_2, cv2.CV_16S, ksize=3, scale=8, delta=0))

# Step 5: Save the images ===================================================================================
# Smoothed
sg_h_list = [
    sg_h_1,
    sg_h_2,
    sg_h_3,
]

sg_v_list = [
    sg_v_1,
    sg_v_2,
    sg_v_3,
]

sb_h_list = [
    sb_h_1,
    sb_h_2,
    sb_h_3,
]

sb_v_list = [
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
    cv2.imwrite(path_sg_v_1, sg_v_1)
    cv2.imwrite(path_sg_v_2, sg_v_2)
    cv2.imwrite(path_sg_v_3, sg_v_3)
    cv2.imwrite(path_sg_v, sg_v)
    cv2.imwrite(path_sg_h_1, sg_h_1)
    cv2.imwrite(path_sg_h_2, sg_h_2)
    cv2.imwrite(path_sg_h_3, sg_h_3)
    cv2.imwrite(path_sb_h, sb_h)
    cv2.imwrite(path_sb_h_1, sb_h_1)
    cv2.imwrite(path_sb_h_2, sb_h_2)
    cv2.imwrite(path_sb_h_3, sb_h_3)
    cv2.imwrite(path_sb_v, sb_v)
    cv2.imwrite(path_sb_v_1, sb_v_1)
    cv2.imwrite(path_sb_v_2, sb_v_2)
    cv2.imwrite(path_sb_v_3, sb_v_3)

# Edge Detection
ec_h_list_1 = [
    ec_h_1,
    ec_h_2,
    ec_h_3,
    ec_h_4,
]

ec_h_list_2 = [
    ec_h_5,
    ec_h_6,
    ec_h_7,
    ec_h_8,
]

ec_v_list_1 = [
    ec_v_1,
    ec_v_2,
    ec_v_3,
    ec_v_4,
]

ec_v_list_2 = [
    ec_v_5,
    ec_v_6,
    ec_v_7,
    ec_v_8,
]

ec_h_a = np.concatenate(ec_h_list_1, 1)
ec_h_b = np.concatenate(ec_h_list_2, 1)
ec_h = np.concatenate([ec_h_a, ec_h_b], 0)
ec_v_a = np.concatenate(ec_v_list_1, 1)
ec_v_b = np.concatenate(ec_v_list_2, 1)
ec_v = np.concatenate([ec_v_a, ec_v_b], 0)


if SAVE_NEW_CANNY:
    cv2.imwrite(path_ec_h, ec_h)
    cv2.imwrite(path_ec_h_1, ec_h_1)
    cv2.imwrite(path_ec_h_2, ec_h_2)
    cv2.imwrite(path_ec_h_3, ec_h_3)
    cv2.imwrite(path_ec_h_4, ec_h_4)
    cv2.imwrite(path_ec_h_5, ec_h_5)
    cv2.imwrite(path_ec_h_6, ec_h_6)
    cv2.imwrite(path_ec_h_7, ec_h_7)
    cv2.imwrite(path_ec_h_8, ec_h_8)
    cv2.imwrite(path_ec_v, ec_v)
    cv2.imwrite(path_ec_v_1, ec_v_1)
    cv2.imwrite(path_ec_v_2, ec_v_2)
    cv2.imwrite(path_ec_v_3, ec_v_3)
    cv2.imwrite(path_ec_v_4, ec_v_4)
    cv2.imwrite(path_ec_v_5, ec_v_5)
    cv2.imwrite(path_ec_v_6, ec_v_6)
    cv2.imwrite(path_ec_v_7, ec_v_7)
    cv2.imwrite(path_ec_v_8, ec_v_8)

el_h_list_1 = [
    el_h_1,
    el_h_2,
    el_h_3,
    el_h_4,
]

el_h_list_2 = [
    el_h_5,
    el_h_6,
    el_h_7,
    el_h_8,
]

el_v_list_1 = [
    el_v_1,
    el_v_2,
    el_v_3,
    el_v_4,
]

el_v_list_2 = [
    el_v_5,
    el_v_6,
    el_v_7,
    el_v_8,
]

el_h_a = np.concatenate(el_h_list_1, 1)
el_h_b = np.concatenate(el_h_list_2, 1)
el_h = np.concatenate([el_h_a, el_h_b], 0)
el_v_a = np.concatenate(el_v_list_1, 1)
el_v_b = np.concatenate(el_v_list_2, 1)
el_v = np.concatenate([el_v_a, el_v_b], 0)

if SAVE_NEW_LAPLACIAN:
    cv2.imwrite(path_el_h, el_h)
    cv2.imwrite(path_el_h_1, el_h_1)
    cv2.imwrite(path_el_h_2, el_h_2)
    cv2.imwrite(path_el_h_3, el_h_3)
    cv2.imwrite(path_el_h_4, el_h_4)
    cv2.imwrite(path_el_h_5, el_h_5)
    cv2.imwrite(path_el_h_6, el_h_6)
    cv2.imwrite(path_el_h_7, el_h_7)
    cv2.imwrite(path_el_h_8, el_h_8)
    cv2.imwrite(path_el_v, el_v)
    cv2.imwrite(path_el_v_1, el_v_1)
    cv2.imwrite(path_el_v_2, el_v_2)
    cv2.imwrite(path_el_v_3, el_v_3)
    cv2.imwrite(path_el_v_4, el_v_4)
    cv2.imwrite(path_el_v_5, el_v_5)
    cv2.imwrite(path_el_v_6, el_v_6)
    cv2.imwrite(path_el_v_7, el_v_7)
    cv2.imwrite(path_el_v_8, el_v_8)

# Step 6: Compute Peak Signal-to-Noise Ratio (PSNR) and compression ratio ===================================
# PSNR - Smoothed Gaussian
psnr_sg_h_1 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_sg_h_1)), 4)
psnr_sg_h_2 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_sg_h_2)), 4)
psnr_sg_h_3 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_sg_h_3)), 4)
psnr_sg_v_1 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_sg_v_1)), 4)
psnr_sg_v_2 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_sg_v_2)), 4)
psnr_sg_v_3 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_sg_v_3)), 4)

# PSNR - Smoothed Bilateral
psnr_sb_h_1 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_sb_h_1)), 4)
psnr_sb_h_2 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_sb_h_2)), 4)
psnr_sb_h_3 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_sb_h_3)), 4)
psnr_sb_v_1 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_sb_v_1)), 4)
psnr_sb_v_2 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_sb_v_2)), 4)
psnr_sb_v_3 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_sb_v_3)), 4)

# PSNR - Edge Detection - Canny
psnr_ec_h_1 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_ec_h_1)), 4)
psnr_ec_h_2 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_ec_h_2)), 4)
psnr_ec_h_3 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_ec_h_3)), 4)
psnr_ec_h_4 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_ec_h_4)), 4)
psnr_ec_h_5 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_ec_h_5)), 4)
psnr_ec_h_6 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_ec_h_6)), 4)
psnr_ec_h_7 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_ec_h_7)), 4)
psnr_ec_h_8 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_ec_h_8)), 4)
psnr_ec_v_1 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_ec_v_1)), 4)
psnr_ec_v_2 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_ec_v_2)), 4)
psnr_ec_v_3 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_ec_v_3)), 4)
psnr_ec_v_4 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_ec_v_4)), 4)
psnr_ec_v_5 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_ec_v_5)), 4)
psnr_ec_v_6 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_ec_v_6)), 4)
psnr_ec_v_7 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_ec_v_7)), 4)
psnr_ec_v_8 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_ec_v_8)), 4)

# PSNR - Edge Detection - Laplacian
# Debug gti_h, el_h_1
psnr_el_h_1 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_el_h_1)), 4)
psnr_el_h_2 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_el_h_2)), 4)
psnr_el_h_3 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_el_h_3)), 4)
psnr_el_h_4 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_el_h_4)), 4)
psnr_el_h_5 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_el_h_5)), 4)
psnr_el_h_6 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_el_h_6)), 4)
psnr_el_h_7 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_el_h_7)), 4)
psnr_el_h_8 = round(cv2.PSNR(cv2.imread(path_gti_h), cv2.imread(path_el_h_8)), 4)
psnr_el_v_1 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_el_v_1)), 4)
psnr_el_v_2 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_el_v_2)), 4)
psnr_el_v_3 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_el_v_3)), 4)
psnr_el_v_4 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_el_v_4)), 4)
psnr_el_v_5 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_el_v_5)), 4)
psnr_el_v_6 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_el_v_6)), 4)
psnr_el_v_7 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_el_v_7)), 4)
psnr_el_v_8 = round(cv2.PSNR(cv2.imread(path_gti_v), cv2.imread(path_el_v_8)), 4)

# Compression Ratio - RAW & GTI
cr_raw_gti_h = round(os.path.getsize(path_gti_h) / os.path.getsize(path_raw_h), 4)
cr_raw_gti_v = round(os.path.getsize(path_gti_v) / os.path.getsize(path_raw_v), 4)

# Compression Ratio - GTI & Smoothed
cr_gti_sg_h_1 = round(os.path.getsize(path_sg_h_1) / os.path.getsize(path_gti_h), 4)
cr_gti_sg_h_2 = round(os.path.getsize(path_sg_h_2) / os.path.getsize(path_gti_h), 4)
cr_gti_sg_h_3 = round(os.path.getsize(path_sg_h_3) / os.path.getsize(path_gti_h), 4)
cr_gti_sg_v_1 = round(os.path.getsize(path_sg_v_1) / os.path.getsize(path_gti_v), 4)
cr_gti_sg_v_2 = round(os.path.getsize(path_sg_v_2) / os.path.getsize(path_gti_v), 4)
cr_gti_sg_v_3 = round(os.path.getsize(path_sg_v_3) / os.path.getsize(path_gti_v), 4)

cr_gti_sb_h_1 = round(os.path.getsize(path_sb_h_1) / os.path.getsize(path_gti_h), 4)
cr_gti_sb_h_2 = round(os.path.getsize(path_sb_h_2) / os.path.getsize(path_gti_h), 4)
cr_gti_sb_h_3 = round(os.path.getsize(path_sb_h_3) / os.path.getsize(path_gti_h), 4)
cr_gti_sb_v_1 = round(os.path.getsize(path_sb_v_1) / os.path.getsize(path_gti_v), 4)
cr_gti_sb_v_2 = round(os.path.getsize(path_sb_v_2) / os.path.getsize(path_gti_v), 4)
cr_gti_sb_v_3 = round(os.path.getsize(path_sb_v_3) / os.path.getsize(path_gti_v), 4)

# Compression Ratio - GTI & Edge Canny
cr_gti_ec_h_1 = round(os.path.getsize(path_ec_h_1) / os.path.getsize(path_gti_h), 4)
cr_gti_ec_h_2 = round(os.path.getsize(path_ec_h_2) / os.path.getsize(path_gti_h), 4)
cr_gti_ec_h_3 = round(os.path.getsize(path_ec_h_3) / os.path.getsize(path_gti_h), 4)
cr_gti_ec_h_4 = round(os.path.getsize(path_ec_h_4) / os.path.getsize(path_gti_h), 4)
cr_gti_ec_h_5 = round(os.path.getsize(path_ec_h_5) / os.path.getsize(path_gti_h), 4)
cr_gti_ec_h_6 = round(os.path.getsize(path_ec_h_6) / os.path.getsize(path_gti_h), 4)
cr_gti_ec_h_7 = round(os.path.getsize(path_ec_h_7) / os.path.getsize(path_gti_h), 4)
cr_gti_ec_h_8 = round(os.path.getsize(path_ec_h_8) / os.path.getsize(path_gti_h), 4)
cr_gti_ec_v_1 = round(os.path.getsize(path_ec_v_1) / os.path.getsize(path_gti_v), 4)
cr_gti_ec_v_2 = round(os.path.getsize(path_ec_v_2) / os.path.getsize(path_gti_v), 4)
cr_gti_ec_v_3 = round(os.path.getsize(path_ec_v_3) / os.path.getsize(path_gti_v), 4)
cr_gti_ec_v_4 = round(os.path.getsize(path_ec_v_4) / os.path.getsize(path_gti_v), 4)
cr_gti_ec_v_5 = round(os.path.getsize(path_ec_v_5) / os.path.getsize(path_gti_v), 4)
cr_gti_ec_v_6 = round(os.path.getsize(path_ec_v_6) / os.path.getsize(path_gti_v), 4)
cr_gti_ec_v_7 = round(os.path.getsize(path_ec_v_7) / os.path.getsize(path_gti_v), 4)
cr_gti_ec_v_8 = round(os.path.getsize(path_ec_v_8) / os.path.getsize(path_gti_v), 4)

# Compression Ratio - GTI & Edge Laplacian
cr_gti_el_h_1 = round(os.path.getsize(path_el_h_1) / os.path.getsize(path_gti_h), 4)
cr_gti_el_h_2 = round(os.path.getsize(path_el_h_2) / os.path.getsize(path_gti_h), 4)
cr_gti_el_h_3 = round(os.path.getsize(path_el_h_3) / os.path.getsize(path_gti_h), 4)
cr_gti_el_h_4 = round(os.path.getsize(path_el_h_4) / os.path.getsize(path_gti_h), 4)
cr_gti_el_h_5 = round(os.path.getsize(path_el_h_5) / os.path.getsize(path_gti_h), 4)
cr_gti_el_h_6 = round(os.path.getsize(path_el_h_6) / os.path.getsize(path_gti_h), 4)
cr_gti_el_h_7 = round(os.path.getsize(path_el_h_7) / os.path.getsize(path_gti_h), 4)
cr_gti_el_h_8 = round(os.path.getsize(path_el_h_8) / os.path.getsize(path_gti_h), 4)
cr_gti_el_v_1 = round(os.path.getsize(path_el_v_1) / os.path.getsize(path_gti_v), 4)
cr_gti_el_v_2 = round(os.path.getsize(path_el_v_2) / os.path.getsize(path_gti_v), 4)
cr_gti_el_v_3 = round(os.path.getsize(path_el_v_3) / os.path.getsize(path_gti_v), 4)
cr_gti_el_v_4 = round(os.path.getsize(path_el_v_4) / os.path.getsize(path_gti_v), 4)
cr_gti_el_v_5 = round(os.path.getsize(path_el_v_5) / os.path.getsize(path_gti_v), 4)
cr_gti_el_v_6 = round(os.path.getsize(path_el_v_6) / os.path.getsize(path_gti_v), 4)
cr_gti_el_v_7 = round(os.path.getsize(path_el_v_7) / os.path.getsize(path_gti_v), 4)
cr_gti_el_v_8 = round(os.path.getsize(path_el_v_8) / os.path.getsize(path_gti_v), 4)

# Step 6: Output the results ===============================================================================+
# ----------------------------------------------------------------------------------------------------------- Show - Information
# Information
print("---------- Informations ----------")
print("raw_h image shape: ", raw_h.shape, " | Type: ", raw_h.dtype, " | Size: ", raw_h.size)
print("raw_v image shape: ", raw_v.shape, " | Type: ", raw_v.dtype, " | Size: ", raw_v.size)
print("gti_h image shape: ", gti_h.shape, " | Type: ", gti_h.dtype, " | Size: ", gti_h.size)
print("gti_v image shape: ", gti_v.shape, " | Type: ", gti_v.dtype, " | Size: ", gti_v.size)
print()
print()
# ----------------------------------------------------------------------------------------------------------- Show - PSNR
print("-------------------- Peak Signal To Noise Ratio (PSNR) --------------------")
# Peak Signal To Noise Ratio - Smoothed
print("Gaussian Blur ------------------------------------------------------------- PSNR")
print("PSNR - GTI X Gaussian Blur - 1 (Horizontal):", psnr_sg_h_1, " dB")
print("PSNR - GTI X Gaussian Blur - 2 (Horizontal):", psnr_sg_h_2, " dB")
print("PSNR - GTI X Gaussian Blur - 3 (Horizontal):", psnr_sg_h_3, " dB")
print("PSNR - GTI X Gaussian Blur - 1 (Vertical):", psnr_sg_v_1, " dB")
print("PSNR - GTI X Gaussian Blur - 2 (Vertical):", psnr_sg_v_2, " dB")
print("PSNR - GTI X Gaussian Blur - 3 (Vertical):", psnr_sg_v_3, " dB")
print()

print("Bilateral Filtering ------------------------------------------------------- PSNR")
print("PSNR - GTI X Bilateral Filtering - 1 (Horizontal):", psnr_sb_h_1, " dB")
print("PSNR - GTI X Bilateral Filtering - 2 (Horizontal):", psnr_sb_h_2, " dB")
print("PSNR - GTI X Bilateral Filtering - 3 (Horizontal):", psnr_sb_h_3, " dB")
print("PSNR - GTI X Bilateral Filtering - 1 (Vertical):", psnr_sb_v_1, " dB")
print("PSNR - GTI X Bilateral Filtering - 2 (Vertical):", psnr_sb_v_2, " dB")
print("PSNR - GTI X Bilateral Filtering - 3 (Vertical):", psnr_sb_v_3, " dB")
print()

# Peak Signal To Noise Ratio - Edge Detection - Canny
print("Canny Edge Detection ------------------------------------------------------ PSNR")
print("PSNR - GTI X Canny - 1 (Horizontal):", psnr_ec_h_1, " dB")
print("PSNR - GTI X Canny - 2 (Horizontal):", psnr_ec_h_2, " dB")
print("PSNR - GTI X Canny - 3 (Horizontal):", psnr_ec_h_3, " dB")
print("PSNR - GTI X Canny - 4 (Horizontal):", psnr_ec_h_4, " dB")
print("PSNR - GTI X Canny - 5 (Horizontal):", psnr_ec_h_5, " dB")
print("PSNR - GTI X Canny - 6 (Horizontal):", psnr_ec_h_6, " dB")
print("PSNR - GTI X Canny - 7 (Horizontal):", psnr_ec_h_7, " dB")
print("PSNR - GTI X Canny - 8 (Horizontal):", psnr_ec_h_8, " dB")
print("PSNR - GTI X Canny - 1 (Vertical):", psnr_ec_v_1, " dB")
print("PSNR - GTI X Canny - 2 (Vertical):", psnr_ec_v_2, " dB")
print("PSNR - GTI X Canny - 3 (Vertical):", psnr_ec_v_3, " dB")
print("PSNR - GTI X Canny - 4 (Vertical):", psnr_ec_v_4, " dB")
print("PSNR - GTI X Canny - 5 (Vertical):", psnr_ec_v_5, " dB")
print("PSNR - GTI X Canny - 6 (Vertical):", psnr_ec_v_6, " dB")
print("PSNR - GTI X Canny - 7 (Vertical):", psnr_ec_v_7, " dB")
print("PSNR - GTI X Canny - 8 (Vertical):", psnr_ec_v_8, " dB")
print()

# Peak Signal To Noise Ratio - Edge Detection - Laplacian
print("Laplacian Edge Detection -------------------------------------------------- PSNR")
print("PSNR - GTI X Laplacian - 1 (Horizontal):", psnr_el_h_1, " dB")
print("PSNR - GTI X Laplacian - 2 (Horizontal):", psnr_el_h_2, " dB")
print("PSNR - GTI X Laplacian - 3 (Horizontal):", psnr_el_h_3, " dB")
print("PSNR - GTI X Laplacian - 4 (Horizontal):", psnr_el_h_4, " dB")
print("PSNR - GTI X Laplacian - 5 (Horizontal):", psnr_el_h_5, " dB")
print("PSNR - GTI X Laplacian - 6 (Horizontal):", psnr_el_h_6, " dB")
print("PSNR - GTI X Laplacian - 7 (Horizontal):", psnr_el_h_7, " dB")
print("PSNR - GTI X Laplacian - 8 (Horizontal):", psnr_el_h_8, " dB")
print("PSNR - GTI X Laplacian - 1 (Vertical):", psnr_el_v_1, " dB")
print("PSNR - GTI X Laplacian - 2 (Vertical):", psnr_el_v_2, " dB")
print("PSNR - GTI X Laplacian - 3 (Vertical):", psnr_el_v_3, " dB")
print("PSNR - GTI X Laplacian - 4 (Vertical):", psnr_el_v_4, " dB")
print("PSNR - GTI X Laplacian - 5 (Vertical):", psnr_el_v_5, " dB")
print("PSNR - GTI X Laplacian - 6 (Vertical):", psnr_el_v_6, " dB")
print("PSNR - GTI X Laplacian - 7 (Vertical):", psnr_el_v_7, " dB")
print("PSNR - GTI X Laplacian - 8 (Vertical):", psnr_el_v_8, " dB")
print()
print()
# ----------------------------------------------------------------------------------------------------------- Show Compression Ratio
print("-------------------- Compression Ratio --------------------")
# Compression Ratio - Scaled
print("Compression Ratio GTI / RAW (Horizontal): ", cr_raw_gti_h)
print("Compression Ratio GTI / RAW (Vertical): ", cr_raw_gti_v)
print()

# Compression Ratio - GTI X Smoothed
print("Gaussian Blur ---------------------------------------------------------- Compression Ratio")
print("Compression Ratio Gaussian Blur / GTI - 1 (Horizontal): ", cr_gti_sg_h_1)
print("Compression Ratio Gaussian Blur / GTI - 2 (Horizontal): ", cr_gti_sg_h_2)
print("Compression Ratio Gaussian Blur / GTI - 3 (Horizontal): ", cr_gti_sg_h_3)
print("Compression Ratio Gaussian Blur / GTI - 1 (Vertical): ", cr_gti_sg_v_1)
print("Compression Ratio Gaussian Blur / GTI - 2 (Vertical): ", cr_gti_sg_v_2)
print("Compression Ratio Gaussian Blur / GTI - 3 (Vertical): ", cr_gti_sg_v_3)
print()

print("Bilateral Filtering ---------------------------------------------------- Compression Ratio")
print("Compression Ratio Bilateral Filtering / GTI - 1 (Horizontal): ", cr_gti_sb_h_1)
print("Compression Ratio Bilateral Filtering / GTI - 2 (Horizontal): ", cr_gti_sb_h_2)
print("Compression Ratio Bilateral Filtering / GTI - 3 (Horizontal): ", cr_gti_sb_h_3)
print("Compression Ratio Bilateral Filtering / GTI - 1 (Vertical): ", cr_gti_sb_v_1)
print("Compression Ratio Bilateral Filtering / GTI - 2 (Vertical): ", cr_gti_sb_v_2)
print("Compression Ratio Bilateral Filtering / GTI - 3 (Vertical): ", cr_gti_sb_v_3)
print()

# Compression Ratio - GTI X Edge Detection - Canny
print("Canny Edge Detection --------------------------------------------------- Compression Ratio")
print("Compression Ratio Canny / GTI - 1 (Horizontal): ", cr_gti_ec_h_1)
print("Compression Ratio Canny / GTI - 2 (Horizontal): ", cr_gti_ec_h_2)
print("Compression Ratio Canny / GTI - 3 (Horizontal): ", cr_gti_ec_h_3)
print("Compression Ratio Canny / GTI - 4 (Horizontal): ", cr_gti_ec_h_4)
print("Compression Ratio Canny / GTI - 5 (Horizontal): ", cr_gti_ec_h_5)
print("Compression Ratio Canny / GTI - 6 (Horizontal): ", cr_gti_ec_h_6)
print("Compression Ratio Canny / GTI - 7 (Horizontal): ", cr_gti_ec_h_7)
print("Compression Ratio Canny / GTI - 8 (Horizontal): ", cr_gti_ec_h_8)
print("Compression Ratio Canny / GTI - 1 (Vertical): ", cr_gti_ec_v_1)
print("Compression Ratio Canny / GTI - 2 (Vertical): ", cr_gti_ec_v_2)
print("Compression Ratio Canny / GTI - 3 (Vertical): ", cr_gti_ec_v_3)
print("Compression Ratio Canny / GTI - 4 (Vertical): ", cr_gti_ec_v_4)
print("Compression Ratio Canny / GTI - 5 (Vertical): ", cr_gti_ec_v_5)
print("Compression Ratio Canny / GTI - 6 (Vertical): ", cr_gti_ec_v_6)
print("Compression Ratio Canny / GTI - 7 (Vertical): ", cr_gti_ec_v_7)
print("Compression Ratio Canny / GTI - 8 (Vertical): ", cr_gti_ec_v_8)
print()

# Compression Ratio - GTI X Edge Detection - Laplacian
print("Laplacian Edge Detection ----------------------------------------------- Compression Ratio")
print("Compression Ratio Laplacian / GTI - 1 (Horizontal): ", cr_gti_el_h_1)
print("Compression Ratio Laplacian / GTI - 2 (Horizontal): ", cr_gti_el_h_2)
print("Compression Ratio Laplacian / GTI - 3 (Horizontal): ", cr_gti_el_h_3)
print("Compression Ratio Laplacian / GTI - 4 (Horizontal): ", cr_gti_el_h_4)
print("Compression Ratio Laplacian / GTI - 5 (Horizontal): ", cr_gti_el_h_5)
print("Compression Ratio Laplacian / GTI - 6 (Horizontal): ", cr_gti_el_h_6)
print("Compression Ratio Laplacian / GTI - 7 (Horizontal): ", cr_gti_el_h_7)
print("Compression Ratio Laplacian / GTI - 8 (Horizontal): ", cr_gti_el_h_8)
print("Compression Ratio Laplacian / GTI - 1 (Vertical): ", cr_gti_el_v_1)
print("Compression Ratio Laplacian / GTI - 2 (Vertical): ", cr_gti_el_v_2)
print("Compression Ratio Laplacian / GTI - 3 (Vertical): ", cr_gti_el_v_3)
print("Compression Ratio Laplacian / GTI - 4 (Vertical): ", cr_gti_el_v_4)
print("Compression Ratio Laplacian / GTI - 5 (Vertical): ", cr_gti_el_v_5)
print("Compression Ratio Laplacian / GTI - 6 (Vertical): ", cr_gti_el_v_6)
print("Compression Ratio Laplacian / GTI - 7 (Vertical): ", cr_gti_el_v_7)
print("Compression Ratio Laplacian / GTI - 8 (Vertical): ", cr_gti_el_v_8)
print()
# ----------------------------------------------------------------------------------------------------------- Show Image

exit()
# ----------------------------------------------------------------------------------------------------------- Show
