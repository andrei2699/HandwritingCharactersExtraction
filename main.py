import os
import shutil
from pathlib import Path

import cv2
import numpy as np


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


def count_mean_character_height(extracted_characters_dict):
    return np.median([character["image"].shape[0] for _, character in sorted(extracted_characters_dict.items())])


def count_average_character_width(extracted_characters_dict):
    character_widths = []
    for x, character in sorted(extracted_characters_dict.items()):
        img = character['image']
        character_widths.append(img.shape[1])

    return np.mean(character_widths)


def display_bounding_box_characters_in_line(line_image, characters_dict):
    for _, character in sorted(characters_dict.items()):
        x, y, w, h = character['coordinates']
        cv2.rectangle(line_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    display_image("line", line_image, (line_image.shape[1], line_image.shape[0]))


def split_too_big_characters(extracted_characters_dict, use_mean=None):
    if use_mean is None:
        average_character_width = count_mean_character_height(extracted_characters_dict)
    else:
        average_character_width = count_average_character_width(extracted_characters_dict)

    if average_character_width == 0:
        average_character_width = 1

    new_extracted_characters_dict = {}

    for _, character in sorted(extracted_characters_dict.items()):
        img = character["image"]
        x, y, w, h = character["coordinates"]

        percent = average_character_width / img.shape[1]

        print(x, w, img.shape[1], average_character_width, percent)

        if percent > 1.5:
            split_point = int(img.shape[1] / 2)

            new_extracted_characters_dict[x] = {
                "image": img[:, :split_point],
                "coordinates": (x, y, split_point, h)
            }
            new_extracted_characters_dict[x + split_point] = {
                "image": img[:, split_point:],
                "coordinates": (x + split_point, y, split_point, h)
            }
        else:
            new_extracted_characters_dict[x] = character

    return new_extracted_characters_dict


def extract_characters(line_image, line_folder):
    create_folder(line_folder)

    copy_image = line_image.copy()

    _, threshold = cv2.threshold(copy_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, opening_kernel, iterations=1)
    # display_image("opening", opening, (copy_image.shape[1], copy_image.shape[0]))

    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extracted_characters_dict = {}

    character_count = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w < 10 or h < 10:
            continue

        crop = line_image[y:y + h, x:x + w]

        # cv2.rectangle(copy_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        extracted_characters_dict[x] = {
            "image": crop,
            "coordinates": (x, y, w, h)
        }

        character_count += 1

    # uncomment if splitting too big characters is wanted
    # extracted_characters_dict = split_too_big_characters(extracted_characters_dict)

    # display_bounding_box_characters_in_line(copy_image, extracted_characters_dict)

    character_count = 0
    for x, character in sorted(extracted_characters_dict.items()):
        img = character["image"]

        save_image(line_folder + "/char" + str(character_count) + ".jpg", img)
        character_count += 1


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

        crop = original_copy[y:y + h, x:x + w]

        center = (x + w // 2, y + h // 2)
        rotation_angle = -90 + angle
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated = cv2.warpAffine(crop, rotation_matrix, (crop.shape[1], crop.shape[0]))

        display_image("line" + str(line_count), rotated, (w, h))

        line_path = folder + '/line' + str(line_count)
        save_image(line_path + '.jpg', rotated)

        rotated = convert_to_gray(rotated)
        rotated = fill_black(rotated, 255)

        extract_characters(rotated, line_path)

        line_count += 1

        # cv2.rectangle(image, (x, y), (x + w, y + h), (channel, 0, 0), 2)
        channel += increment


def delete_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)


if __name__ == '__main__':
    # image = read_image('data/0002_3.jpg')
    image = read_image('data/0001_3.jpg')
    lines_output_folder = 'output'

    delete_folder(lines_output_folder)

    extract_lines(image, lines_output_folder)

    # display_image("bounding box", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
