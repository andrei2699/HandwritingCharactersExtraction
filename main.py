import cv2
import numpy as np
import os
from pathlib import Path


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_image(path, image):
    parent = Path(path).parent.absolute()
    create_folder(parent)

    cv2.imwrite(path, image)


def display_image(name, image, size=None):
    if size is None:
        size = image.shape
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, size[0], size[1])
    cv2.imshow(name, image)


def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def extract_line_contours(image):
    gray_image = convert_to_gray(image)
    # display_image("grayscale", gray_image)

    _, threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # display_image("Threshold", threshold)

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erode = cv2.erode(threshold, erode_kernel, iterations=1)

    # display_image("Erode", erode)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 4))
    dilate = cv2.dilate(erode, dilate_kernel, iterations=2)

    # display_image("dilate", dilate)

    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 10))
    opening = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, opening_kernel, iterations=2)

    # display_image("opening", opening)

    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def read_image(path):
    return cv2.imread(path)


def get_rotated_rectangle_from_contour(contour):
    (x, y), (w, h), angle = cv2.minAreaRect(contour)
    box = cv2.boxPoints(((x, y), (w, h), angle))
    box = np.intp(box)

    return box, angle


def fill_black(image, color):
    new_image = image.copy()

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x, y] == 0:
                new_image[x, y] = color
    return new_image


def extract_characters(line_image, line_folder):
    create_folder(line_folder)


# TODO: character extraction
# tips: calculate the average width of the characters and use it to extract the characters that are close to each other


def extract_lines(image, folder):
    line_contours = extract_line_contours(image)

    channel = 0
    increment = 255 // (len(line_contours) == 0 and 1 or len(line_contours))

    line_count = 0
    for contour in line_contours:
        box, angle = get_rotated_rectangle_from_contour(contour)
        original_copy = image.copy()

        # cv2.drawContours(original_copy, [box], 0, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(contour)

        # original_copy = convert_to_gray(original_copy)

        crop = original_copy[y:y + h, x:x + w]

        center = (x + w // 2, y + h // 2)
        rotation_angle = -90 + angle
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated = cv2.warpAffine(crop, rotation_matrix, (crop.shape[1], crop.shape[0]))

        # rotated = fill_black(rotated, 255)

        display_image("line" + str(line_count), rotated, (w, h))

        line_path = folder + '/line' + str(line_count)
        save_image(line_path + '.jpg', rotated)

        extract_characters(rotated, line_path)

        line_count += 1

        # cv2.rectangle(image, (x, y), (x + w, y + h), (channel, 0, 0), 2)
        channel += increment


if __name__ == '__main__':
    # image = read_image('data/0002_3.jpg')
    image = read_image('data/0001_3.jpg')
    lines_output_folder = 'output'

    extract_lines(image, lines_output_folder)

    # display_image("bounding box", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
