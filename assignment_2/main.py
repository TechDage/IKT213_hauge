import cv2
import numpy as np

#from keras.models import load_model
#from keras.preprocessing.image import ImageDataGenerator
#img = cv2.imread('iris-1.jpg',0)

def padding(image, border_width):
    padded = cv2.copyMakeBorder(image, border_width, border_width,
    border_width, border_width, cv2.BORDER_REFLECT)
    cv2.imwrite("padding.png", padded)
    return padded

def crop(image, x_0, x_1, y_0, y_1):
    cropped = image[y_0:y_1, x_0:x_1]
    cv2.imwrite("crop.png", cropped)
    return cropped

def resize(image, width, height):
    resized = cv2.resize(image, (width, height))
    cv2.imwrite("resize.png", resized)
    return resized

def copy(image, emptyPictureArray):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = image[y, x, c]
    cv2.imwrite("copy.png", emptyPictureArray)
    return emptyPictureArray

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("grayscale.png", gray)
    return gray

def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("hsv.png", hsv_image)
    return hsv_image

def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for c in range(channels):

                original_value = image[y, x, c]
                original_value = int(original_value)
                new_value = original_value + hue

                if new_value > 255:
                    new_value = 255
                if new_value < 0:
                    new_value = 0
                emptyPictureArray[y, x, c] = new_value
#med hue verdien i main med -250 så får man helt svart bilde i kontrast til 260, hvor man da får et helt hvit bilde
    cv2.imwrite("hue_shifted.png", emptyPictureArray)
    return emptyPictureArray

def smoothing(image):
    smoothed = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)
    cv2.imwrite("smoothing.png", smoothed)
    return smoothed

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite("rotation90.png", rotated)
    elif rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imwrite("rotation180.png", rotated)
    return rotated

def main():
    image = cv2.imread("iris-1.png")
    height, width, channels = image.shape

    padding(image, 100)
    crop(image, 200, width - 130, 200, height - 130)
    resize(image, 200, 200)

    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
    copy(image, emptyPictureArray)

    grayscale(image)
    hsv(image)

    emptyPictureArray2 = np.zeros((height, width, 3), dtype=np.uint8)
    hue_shifted(image, emptyPictureArray2, 50)

    smoothing(image)
    rotation(image, 90)
    rotation(image, 180)

if __name__ == "__main__":
    main()