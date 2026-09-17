import cv2
import numpy as np


def sobel_edge_detection(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    sobel = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)
    normalized = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imwrite("sobel.png", normalized)
    return normalized

def canny_edge_detection(image, threshold_1, threshold_2):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    edges = cv2.Canny(image=img_blur, threshold1=threshold_1, threshold2=threshold_2)

    cv2.imwrite("canny.png", edges)
    return edges

def template_match(image, template):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    threshold = 0.9
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite("matchingdone.png", image)
    return image

def resize(image, scale_factor: int, up_or_down: str):
    resized = image

    for i in range(scale_factor):
        rows, cols, _channels = map(int, resized.shape)

        if up_or_down == "up":
            resized = cv2.pyrUp(resized, dstsize=(2 * cols, 2 * rows))
        elif up_or_down == "down":
            resized = cv2.pyrDown(resized, dstsize=(cols // 2, rows // 2))

    if up_or_down == "up":
        cv2.imwrite("resizeup.png", resized)
    elif up_or_down == "down":
        cv2.imwrite("resizedown.png", resized)

    return resized

def main():
    image = cv2.imread("lambo.png")
    sobel_edge_detection(image)
    canny_edge_detection(image, 50, 50)
    resize(image, 2, "up")
    resize(image, 2, "down")
    shapes = cv2.imread("shapes-1.png")
    template = cv2.imread("shapes_template.jpg")
    template_match(shapes, template)

if __name__ == "__main__":
    main()
