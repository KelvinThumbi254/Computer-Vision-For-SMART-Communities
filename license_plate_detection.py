import cv2
import numpy as np


# Define an image preprocessing code
def preprocess(gray):
    """
    Transform the shape of grayscale objects (pre-processing)
    :param gray:
    :return:
    """
    # Gaussian smoothing
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0)#, 0 cv2.BORDER_DEFAULT)

    # Median filter
    median = cv2.medianBlur(gaussian, 5)

    # Sobel operator，process the edges which is a convolution
    # x：[-1, 0, +1, -2, 0, +2, -1, 0, +1]
    # y：[-1, -2, -1, 0, 0, 0, +1, +2, +1]
    sobel = cv2.Sobel(median, cv2.CV_64F, dx=1, dy=0, ksize=3)
    # The type is converted to unit8
    sobel = np.uint8(np.absolute(sobel))

    # Binaryzation
    ret, binary = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)

    # dilation and erosion
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
    # Expand once so the outline stands out
    dilation = cv2.dilate(binary, element2, iterations=1)
    # Corrode once, remove details
    erosion = cv2.erode(dilation, element1, iterations=2)
    # Expand it again to make it more visible
    dilation2 = cv2.dilate(erosion, element2, iterations=5)
    # Corrode once, remove details
    erosion2 = cv2.erode(dilation2, element1, iterations=4)

    # cv2.imshow('gray', gray)
    # cv2.imshow('gaussian', gaussian)
    # cv2.imshow('median', median)
    # cv2.imshow('sobel', sobel)
    # cv2.imshow('binary', binary)
    # cv2.imshow('dilation', dilation)
    # cv2.imshow('erosion', erosion)
    # cv2.imshow('dilation2', dilation2)
    # cv2.imshow('erosion2', erosion2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return erosion2

img = cv2.imread("car.jpg", 0)
preprocess(img)


# Do a license plate area search
def find_plate_number_region(img):
    """
    Look for the outline of a possible license plate area
    :param img:
    :return:
    """
    # Find contours (img: original image, contours: rectangular coordinate points, hierarchy: image hierarchy)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find rectangle
    max_ratio = -1
    max_box = None
    ratios = []
    number = 0
    for i in range(len(contours)):
        cnt = contours[i]  # Coordinates of the current profile

        # Calculated contour area
        area = cv2.contourArea(cnt)
        # Filter out areas that are too small
        if area < 10000:
            continue

        # Find the smallest rectangle
        rect = cv2.minAreaRect(cnt)

        # The four coordinates of the rectangle (the order varies, but it must be a circular order of bottom left, top left, top right, bottom right (unknown starting point))
        box = cv2.boxPoints(rect)
        # Convert to the long type
        box = np.int64(box)

        # Calculate length, width and height
        # Calculate the length of the first edge
        a = abs(box[0][0] - box[1][0])
        b = abs(box[0][1] - box[1][1])
        d1 = np.sqrt(a ** 2 + b ** 2)
        # Calculate the length of the second side
        c = abs(box[1][0] - box[2][0])
        d = abs(box[1][1] - box[2][1])
        d2 = np.sqrt(c ** 2 + d ** 2)
        # Let the minimum be the height and the maximum the width
        height = int(min(d1, d2))
        weight = int(max(d1, d2))

        # calculate area
        area2 = height * weight

        # The difference between the two areas must be within a certain range
        r = np.absolute((area2 - area) / area)
        if r > 0.6:
            continue

        ratio = float(weight) / float(height)
        print((box, height, weight, area, area2, r, ratio, rect[-1]))
        cv2.drawContours(img, [box], 0, 255, 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 实In practice, the ratio should be about 3, but our photos are not standard
        # he measured width to height should be between 2 and 5.5
        if ratio > max_ratio:
            max_box = box
            max_ratio = ratio

        if ratio > 5.5 or ratio < 2:
            continue

        number += 1
        ratios.append((box, ratio))

    # Output data based on the number of image matrices found
    print("Total found :{} possible regions!!".format(number))
    if number == 1:
        # Direct return
        return ratios[0][0]
    elif number > 1:
        # Take the median value without thinking too much (and filter it)
        # The actual requirements are more stringent
        filter_ratios = list(filter(lambda t: 2.7 <= t[1] <= 5.0, ratios))
        size_filter_ratios = len(filter_ratios)

        if size_filter_ratios == 1:
            return filter_ratios[0][0]
        elif size_filter_ratios > 1:
            # Get the median
            ratios1 = [filter_ratios[i][1] for i in range(size_filter_ratios)]
            ratios1 = list(zip(range(size_filter_ratios), ratios1))
            # Sorting data
            ratios1 = sorted(ratios1, key=lambda t: t[1])
            # Get data for the median value
            idx = ratios1[size_filter_ratios // 2][0]
            return filter_ratios[idx][0]
        else:
            # Get the maximum
            ratios1 = [ratios[i][1] for i in range(number)]
            ratios1 = list(zip(range(number), ratios1))
            # Sorting data
            ratios1 = sorted(ratios1, key=lambda t: t[1])
            # Get the maximum value
            idx = ratios1[-1][0]
            return filter_ratios[idx][0]
    else:
        # Direct return to maximum
        print("Return directly to the region closest to the scale...")
        return max_box


# For license plate interception
def cut(img_or_img_path):
    """
    Intercept the license plate area
    :param img_or_img_path:
    :return:
    """
    if isinstance(img_or_img_path, str):
        img = cv2.imread(img_or_img_path)
    else:
        img = img_or_img_path

    # Gets the height and width of the image
    rows, cols, _ = img.shape

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Image preprocessing --> The license plate area is clearly displayed
    dilation = preprocess(gray)

    # Find the license plate area (assuming there will only be one)
    box = find_plate_number_region(dilation)

    # Returns the image corresponding to the region
    # Due to we do not know the order of the points, so I sort the coordinates of the points on the left
    ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
    xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
    ys_sorted_index = np.argsort(ys)
    xs_sorted_index = np.argsort(xs)

    # Gets the coordinates on x
    x1 = box[xs_sorted_index[0], 0]
    x1 = x1 if x1 > 0 else 0
    x2 = box[xs_sorted_index[3], 0]
    x2 = cols if x2 > cols else x2

    # Get the coordinates on y
    y1 = box[ys_sorted_index[0], 1]
    y1 = y1 if y1 > 0 else 0
    y2 = box[ys_sorted_index[3], 1]
    y2 = rows if y2 > rows else y2

    # Intercept image
    img_plate = img[y1:y2, x1:x2]

    return img_plate

path = 'car.jpg'
cut_img = cut(path)
print(cut_img.shape)
cv2.imwrite(f'plat_{path}', cut_img)

# visualization
cv2.imshow('image', cv2.imread(path))
cv2.imshow('plat', cut_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
