import cv2
import numpy as np

#from keras.models import load_model
#from keras.preprocessing.image import ImageDataGenerator
#img = cv2.imread('iris-1.jpg',0)
def print_image_information(image):
    height, width, channels = image.shape
    size = image.size
    data_type = image.dtype
    print(f"height: {height}")
    print(f"width: {width}")
    print(f"channels: {channels}")
    print(f"size: {size}")
    print(f"data type: {data_type}")

def save_camera_information():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error no camera")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()

    with open("camera_outputs.txt", "w") as f:
        f.write(f"fps: {int(fps)}\n")
        f.write(f"height: {int(height)}\n")
        f.write(f"width: {int(width)}\n")
def main():
    image = cv2.imread("iris-1.jpg")
    print_image_information(image)
    save_camera_information()
if __name__ == "__main__":
    main()
