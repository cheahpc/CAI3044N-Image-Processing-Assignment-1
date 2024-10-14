import cv2
import numpy as np
import rawpy


# Converter +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Converter:
    # Enumerate
    RAW2BGR: int
    RGB2GRAY: int
    RGB2BGR: int
    BGR2GRAY: int
    BGR2RGB: int
    GRAY2BGR: int
    GRAY2RGB: int

    # Convert Raw to BGR
    def raw2bgr(imgSet, cameraWB=True):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.cvtColor(
                rawpy.imread(imgSet).postprocess(use_camera_wb=cameraWB),
                cv2.COLOR_RGB2BGR,
            )
        else:
            return [
                cv2.cvtColor(
                    rawpy.imread(img).postprocess(use_camera_wb=cameraWB),
                    cv2.COLOR_RGB2BGR,
                )
                for img in imgSet
            ]

    # Convert RGB to Gray
    def rgb2gray(imgSet):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.cvtColor(imgSet, cv2.COLOR_RGB2GRAY)
        else:
            return [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgSet]

    # Convert RGB to BGR
    def rgb2bgr(imgSet):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.cvtColor(imgSet, cv2.COLOR_RGB2BGR)
        else:
            return [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgSet]

    # Convert BGR to Gray
    def bgr2gray(imgSet):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.cvtColor(imgSet, cv2.COLOR_BGR2GRAY)
        else:
            return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgSet]

    # Convert BGR to RGB
    def bgr2rgb(imgSet):
        # Convert BGR image to RGB
        return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgSet]

    # Convert Gray to BGR
    def gray2bgr(imgSet):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.cvtColor(imgSet, cv2.COLOR_GRAY2BGR)
        else:
            return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in imgSet]

    # Convert Gray to RGB
    def gray2rgb(imgSet):
        # Convert Gray image to RGB
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in imgSet]


# Manipulate image ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Image:
    # Read image set
    def readSet(imgSet):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.imread(imgSet)
        else:
            return [cv2.imread(img) for img in imgSet]

    # Save image to TIFF
    def saveSet(imgSet, fileName):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            cv2.imwrite(fileName, imgSet)
        elif len(imgSet) != len(fileName):
            print("Error: Image length mismatch")
        else:
            for i in range(len(imgSet)):
                cv2.imwrite(fileName[i], imgSet[i])
        return None

    # Scale images
    def scaleSetBy(imgSet, scaleFactor=0.1):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.resize(imgSet, None, fx=scaleFactor, fy=scaleFactor)
        else:
            return [
                cv2.resize(img, None, fx=scaleFactor, fy=scaleFactor) for img in imgSet
            ]

    def scaleSetToHeight(imgSet, height):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.resize(
                imgSet, (int(imgSet.shape[1] * height / imgSet.shape[0]), height)
            )
        else:
            return [
                cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height))
                for img in imgSet
            ]

    def scaleSetToWidth(imgSet, width):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.resize(
                imgSet, (width, int(imgSet.shape[0] * width / imgSet.shape[1]))
            )
        else:
            return [
                cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
                for img in imgSet
            ]

    # Concatenate images
    def concat(img1, img2, axis=1):
        # Concatenate two images, axis=1 for horizontal, axis=0 for vertical
        # Get new height and width
        newHeight = max(img1.shape[0], img2.shape[0])
        newWidth = max(img1.shape[1], img2.shape[1])
        # Concatenate two images with flexible size
        return np.concatenate(
            [
                cv2.copyMakeBorder(
                    img,
                    0,
                    newHeight - img.shape[0],
                    0,
                    newWidth - img.shape[1],
                    cv2.BORDER_CONSTANT,
                )
                for img in [img1, img2]
            ],
            axis=axis,
        )

    def concatSet(imgSet, axis=1):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            print("Error: Image set is not array")
            return None
        elif len(imgSet) == 0:
            print("Error: Image set is empty")
            return None
        else:
            # Make all image in the array with same channel size
            for i in range(len(imgSet)):
                if len(imgSet[i].shape) == 2:
                    imgSet[i] = cv2.cvtColor(imgSet[i], cv2.COLOR_GRAY2BGR)
                    
            # Concatenate all images in the set
            return np.concatenate(imgSet, axis=axis)

    # Show all images
    def showSet(imgSet, name="image"):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            cv2.imshow(name, imgSet)
            return None
        elif len(imgSet) == 0:
            print("Error: Image set is empty")
            return None
        else:
            for i in range(len(imgSet)):
                cv2.imshow(name + str(i + 1), imgSet[i])


# Smooth Filter +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. Average Filter
# 2. Box Filter
# 3. Gaussian Filter
# 4. Median Filter
# 5. Bilateral Filter
# 6. Non-Local Means Filter
# 7. Custom Kernel Filter
class SmoothFilter:

    # Create struct for filter type
    def applyFilter(
        imgSet,
        filterType,
        kernelSizeX=5,
        kernelSizeY=5,
        kSize=5,
        sigmaX=0,
        sigmaY=0,
        d=9,
        sigmaColor=75,
        sigmaSpace=75,
        h=10,
        searchWindowSize=20,
    ):

        # Check imgSet Instance
        if not isinstance(imgSet, list):
            if filterType == 1:  # Average Filter
                return cv2.blur(imgSet, (kernelSizeX, kernelSizeY))
            elif filterType == 2:  # Box Filter
                return cv2.boxFilter(imgSet, -1, (kernelSizeX, kernelSizeY))
            elif filterType == 3:  # Gaussian Filter
                return cv2.GaussianBlur(
                    imgSet, (kernelSizeX, kernelSizeY), sigmaX=sigmaX, sigmaY=sigmaY
                )
            elif filterType == 4:  # Median Filter
                return cv2.medianBlur(imgSet, kSize)
            elif filterType == 5:  # Bilateral Filter
                return cv2.bilateralFilter(imgSet, d, sigmaColor, sigmaSpace)
            elif filterType == 6:  # Non-Local Means Filter
                return cv2.fastNlMeansDenoising(imgSet, None, h, searchWindowSize)
            elif filterType == 7:  # Custom Kernel Filter
                kernel = np.ones((kSize, kSize), np.float32) / (kSize * kSize)
                return cv2.filter2D(imgSet, -1, kernel)
            else:
                print("Error: Filter type not found")
                return None
        else:
            if filterType == 1:  # Average Filter
                return [cv2.blur(img, (kernelSizeX, kernelSizeY)) for img in imgSet]
            elif filterType == 2:  # Box Filter
                return [
                    cv2.boxFilter(img, -1, (kernelSizeX, kernelSizeY)) for img in imgSet
                ]
            elif filterType == 3:  # Gaussian Filter
                return [
                    cv2.GaussianBlur(
                        img, (kernelSizeX, kernelSizeY), sigmaX=sigmaX, sigmaY=sigmaY
                    )
                    for img in imgSet
                ]
            elif filterType == 4:  # Median Filter
                return [cv2.medianBlur(img, kSize) for img in imgSet]
            elif filterType == 5:  # Bilateral Filter
                return [
                    cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
                    for img in imgSet
                ]
            elif filterType == 6:  # Non-Local Means Filter
                return [
                    cv2.fastNlMeansDenoising(img, None, h, searchWindowSize)
                    for img in imgSet
                ]
            elif filterType == 7:  # Custom Kernel Filter
                kernel = np.ones((kSize, kSize), np.float32) / (kSize * kSize)
                return [cv2.filter2D(img, -1, kernel) for img in imgSet]
            else:
                print("Error: Filter type not found")
                return None

    def applyAverageFilter(imgSet, kernelSizeX=5, kernelSizeY=5):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.blur(imgSet, (kernelSizeX, kernelSizeY))
        else:
            return [cv2.blur(img, (kernelSizeX, kernelSizeY)) for img in imgSet]

    def applyBoxFilter(imgSet, kernelSizeX=5, kernelSizeY=5):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.boxFilter(imgSet, -1, (kernelSizeX, kernelSizeY))
        else:
            return [
                cv2.boxFilter(img, -1, (kernelSizeX, kernelSizeY)) for img in imgSet
            ]

    def applyGaussianBlurFilter(imgSet, kernelSizeX=5, kernelSizeY=5, sigmaX=0, sigmaY=0):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.GaussianBlur(imgSet, (kernelSizeX, kernelSizeY), sigmaX=sigmaX, sigmaY=sigmaY)
        else:
            return [
                cv2.GaussianBlur(img, (kernelSizeX, kernelSizeY), sigmaX =sigmaX, sigmaY = sigmaY) for img in imgSet
            ]

    def applyMedianBlurFilter(imgSet, kernelSize=5):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.medianBlur(imgSet, kernelSize)
        else:
            return [cv2.medianBlur(img, kernelSize) for img in imgSet]

    def applyBilateralFilter(imgSet, d=9, sigmaColor=75, sigmaSpace=75):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.bilateralFilter(imgSet, d, sigmaColor, sigmaSpace)
        else:
            return [
                cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace) for img in imgSet
            ]

    def applyNonLocalMeansFilter(imgSet, h=10, searchWindowSize=20):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.fastNlMeansDenoising(imgSet, None, h, searchWindowSize)
        else:
            return [
                cv2.fastNlMeansDenoising(img, None, h, searchWindowSize)
                for img in imgSet
            ]

    def applyCustomKernelFilter(imgSet, kSize):
        kernel = np.ones((kSize, kSize), np.float32) / (kSize * kSize)
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.filter2D(imgSet, -1, kernel)
        else:
            return [cv2.filter2D(img, -1, kernel) for img in imgSet]


# Edge Detection Filter ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. Canny Edge Detection
# 2. Sobel Edge Detection
# 3. Laplacian Edge Detection
# 4. Scharr Edge Detection
# 5. Prewitt Edge Detection
class EdgeFilter:
    def applyCanny(imgSet, CannyParam):
        # Threshold 1 and 2 are used to detect strong and weak edges(minVal = 0, maxVal = 255)
        # ApertureSize is the size of Sobel kernel used to find image gradients
        # L2gradient is a flag to specify the equation for finding gradient magnitude
        # Check if image is array or not

        cannySet = []
        if not isinstance(imgSet, list):
            for j in range(len(CannyParam)):
                cannySet.append(
                    cv2.Canny(
                        imgSet,
                        CannyParam[j]["tresh1"],
                        CannyParam[j]["tresh2"],
                        apertureSize=CannyParam[j]["apertureSize"],
                        L2gradient=CannyParam[j]["L2gradient"],
                    )
                )
        else:
            for i in range(len(imgSet)):
                cannySet.append([])
                for j in range(len(CannyParam)):
                    cannySet[i].append(
                        cv2.Canny(
                            imgSet[i],
                            CannyParam[j]["tresh1"],
                            CannyParam[j]["tresh2"],
                            apertureSize=CannyParam[j]["apertureSize"],
                            L2gradient=CannyParam[j]["L2gradient"],
                        )
                    )

        return cannySet


class Compute:
    def peakSignalToNoiseRatio(img1, img2, decimal=4):
        # Check if image is array or not
        if not isinstance(img1, list) and not isinstance(img2, list):
            return cv2.PSNR(img1, img2)
        elif not isinstance(img1, list) or not isinstance(img2, list):
            print("Error: Image type mismatch")
        elif len(img1) != len(img2):
            print("Error: Image length mismatch")
        else:
            # Compute PSNR
            psnr = [None] * len(img1)
            for i in range(len(img1)):
                psnr[i] = round(cv2.PSNR(img1[i], img2[i]), decimal)

            return psnr
        return None

    def psnrToString(psnr, imgSetA, imgSetB):
        # Check if PSNR is array or not
        if not isinstance(psnr, list):
            print("PSNR: " + "imgSetA" + " | " + "imgSetB" + " = " + str(psnr))
        else:
            for i in range(len(psnr)):
                print(
                    "PSNR: "
                    + imgSetA[i].split("/")[-1]
                    + " | "
                    + imgSetB[i].split("/")[-1]
                    + " = "
                    + str(psnr[i])
                )

    def compressionRatio(img1, img2, decimal=4):
        # Check if image is array or not
        if not isinstance(img1, list) and not isinstance(img2, list):
            return round((img1.nbytes / img2.nbytes), decimal)
        elif not isinstance(img1, list) or not isinstance(img2, list):
            print("Error: Image type mismatch")
        elif len(img1) != len(img2):
            print("Error: Image length mismatch")
        else:
            # Compute Compression Ratio
            cr = [None] * len(img1)
            for i in range(len(img1)):
                cr[i] = round((img1[i].nbytes / img2[i].nbytes), decimal)

            return cr
        return None

    def crToString(cr, imgSetA, imgSetB):
        # Check if CR is array or not
        if not isinstance(cr, list):
            print("CR: " + "imgSetA" + " | " + "imgSetB" + " = " + str(cr))
        else:
            for i in range(len(cr)):
                print(
                    "CR: "
                    + imgSetA[i].split("/")[-1]
                    + " | "
                    + imgSetB[i].split("/")[-1]
                    + " = "
                    + str(cr[i])
                )
