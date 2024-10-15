import cv2
import numpy as np
import rawpy


# Converter +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Converter:

    def getRedChannel(imgSet):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            imgSet[:, :, 1] = 0
            imgSet[:, :, 2] = 0
            return imgSet
        else:
            for i in range(len(imgSet)):
                imgSet[i][:, :, 1] = 0
                imgSet[i][:, :, 2] = 0
            return imgSet

    def getGreenChannel(imgSet):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            imgSet[:, :, 0] = 0
            imgSet[:, :, 2] = 0
            return imgSet
        else:
            for i in range(len(imgSet)):
                imgSet[i][:, :, 0] = 0
                imgSet[i][:, :, 2] = 0
            return imgSet

    def getBlueChannel(imgSet):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            imgSet[:, :, 0] = 0
            imgSet[:, :, 1] = 0
            return imgSet
        else:
            for i in range(len(imgSet)):
                imgSet[i][:, :, 0] = 0
                imgSet[i][:, :, 1] = 0
            return imgSet

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
    def readSet(imgSet, gray=False):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            # return cv2.imread(imgSet) if not gray else cv2.imread(imgSet, cv2.IMREAD_GRAYSCALE)
            return cv2.imread(imgSet) if not gray else cv2.imread(imgSet, 0)
        else:
            return [cv2.imread(img) for img in imgSet] if not gray else [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in imgSet]

    # Save image to TIFF
    def saveSet(imgSet, fileName):
        if not isinstance(imgSet, list) and not isinstance(fileName, list):
            cv2.imwrite(fileName, imgSet)
        elif len(imgSet) != len(fileName):
            print("Error: Image length mismatch")
        else:
            for i in range(len(imgSet)):
                cv2.imwrite(fileName[i], imgSet[i])
        return None

    def getImageType(imgSet):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return imgSet.dtype
        else:
            return [img.dtype for img in imgSet]

    # Scale images
    def scaleSetBy(imgSet, scaleFactor=0.1):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.resize(imgSet, None, fx=scaleFactor, fy=scaleFactor)
        else:
            return [cv2.resize(img, None, fx=scaleFactor, fy=scaleFactor) for img in imgSet]

    def scaleSetToHeight(imgSet, height):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.resize(imgSet, (int(imgSet.shape[1] * height / imgSet.shape[0]), height))
        else:
            return [cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height)) for img in imgSet]

    def scaleSetToWidth(imgSet, width):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.resize(imgSet, (width, int(imgSet.shape[0] * width / imgSet.shape[1])))
        else:
            return [cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1]))) for img in imgSet]

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
                return cv2.GaussianBlur(imgSet, (kernelSizeX, kernelSizeY), sigmaX=sigmaX, sigmaY=sigmaY)
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
                return [cv2.boxFilter(img, -1, (kernelSizeX, kernelSizeY)) for img in imgSet]
            elif filterType == 3:  # Gaussian Filter
                return [cv2.GaussianBlur(img, (kernelSizeX, kernelSizeY), sigmaX=sigmaX, sigmaY=sigmaY) for img in imgSet]
            elif filterType == 4:  # Median Filter
                return [cv2.medianBlur(img, kSize) for img in imgSet]
            elif filterType == 5:  # Bilateral Filter
                return [cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace) for img in imgSet]
            elif filterType == 6:  # Non-Local Means Filter
                return [cv2.fastNlMeansDenoising(img, None, h, searchWindowSize) for img in imgSet]
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
            return [cv2.boxFilter(img, -1, (kernelSizeX, kernelSizeY)) for img in imgSet]

    def applyGaussianBlurFilter(imgSet, kernelSize=5, sigmaX=0, sigmaY=0):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.GaussianBlur(imgSet, (kernelSize,kernelSize), sigmaX=sigmaX, sigmaY=sigmaY)
        else:
            return [cv2.GaussianBlur(img, (kernelSize, kernelSize), sigmaX=sigmaX, sigmaY=sigmaY) for img in imgSet]

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
            return [cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace) for img in imgSet]

    def applyNonLocalMeansFilter(imgSet, h=10, searchWindowSize=20):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.fastNlMeansDenoising(imgSet, None, h, searchWindowSize)
        else:
            return [cv2.fastNlMeansDenoising(img, None, h, searchWindowSize) for img in imgSet]

    def applyCustomKernelFilter(imgSet, kSize):
        kernel = np.ones((kSize, kSize), np.float32) / (kSize * kSize)
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.filter2D(imgSet, -1, kernel)
        else:
            return [cv2.filter2D(img, -1, kernel) for img in imgSet]


# Edge Detection Filter ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1. Canny Edge Detection
# 2. Laplacian Edge Detection
class EdgeFilter:
    def applyCanny(imgSet, CannyParam):
        # Threshold 1 and 2 are used to detect strong and weak edges(minVal = 0, maxVal = 255)
        # ApertureSize is the size of Sobel kernel used to find image gradients
        # L2gradient is a flag to specify the equation for finding gradient magnitude

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
        elif not isinstance(CannyParam, list):
            for i in range(len(imgSet)):
                cannySet.append(
                    cv2.Canny(
                        imgSet[i],
                        CannyParam["tresh1"],
                        CannyParam["tresh2"],
                        apertureSize=CannyParam["apertureSize"],
                        L2gradient=CannyParam["L2gradient"],
                    )
                )
        elif not isinstance(imgSet, list) and not isinstance(CannyParam, list):
            cannySet.append(
                cv2.Canny(
                    imgSet,
                    CannyParam["tresh1"],
                    CannyParam["tresh2"],
                    apertureSize=CannyParam["apertureSize"],
                    L2gradient=CannyParam["L2gradient"],
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

    def applyLaplacian(imgSet):
        # Check if image is array or not
        if not isinstance(imgSet, list):
            return cv2.Laplacian(imgSet, cv2.CV_64F)
        else:
            return [cv2.Laplacian(img, cv2.CV_64F) for img in imgSet]


class Compute:
    def peakSignalToNoiseRatio(imgSrc, imgProcessed, decimal=4):
        # Check if image is array or not
        if not isinstance(imgSrc, list) and not isinstance(imgProcessed, list):
            if imgSrc.size != imgProcessed.size:
                print("Error: Image size mismatch")
                return None
            return cv2.PSNR(imgSrc, imgProcessed)
        elif not isinstance(imgSrc, list) or not isinstance(imgProcessed, list):
            print("Error: Image type mismatch")
        elif len(imgSrc) != len(imgProcessed):
            print("Error: Image length mismatch")
        else:
            # Compute PSNR
            psnr = [] 
            for i in range(len(imgSrc)):
                psnr[i] = round(cv2.PSNR(imgSrc[i], imgProcessed[i]), decimal)

            return psnr
        return None

    def compressionRatio(imgSrc, imgProcessed, decimal=4):
        # Check if image is array or not
        if not isinstance(imgSrc, list) and not isinstance(imgProcessed, list):
            return round(imgSrc.size / imgProcessed.size, decimal)
        elif not isinstance(imgSrc, list) or not isinstance(imgProcessed, list):
            print("Error: Image type mismatch")
        elif len(imgSrc) != len(imgProcessed):
            print("Error: Image length mismatch")
        else:
            # Compute Compression Ratio
            cr = []
            for i in range(len(imgSrc)):
                cr.append(round((imgSrc[i].size / imgProcessed[i].size), decimal))
            return cr
