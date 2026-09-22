import cv2
import numpy as np

def harris(reference_image):
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    res = np.hstack((centroids, corners))
    res = np.intp(res)
    reference_image[res[:, 1], res[:, 0]] = [0, 0, 255]
    reference_image[res[:, 3], res[:, 2]] = [0, 255, 0]
    reference_image[dst > 0.5 * dst.max()] = [0, 0, 255]
    cv2.imwrite("harris.png", reference_image)
    return reference_image

def alignment_sift(image_to_align, reference_image, max_features, good_match_percent):
    gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < good_match_percent * n.distance:
            good.append(m)
    if len(good) > max_features:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        height, width, channels = reference_image.shape
        aligned = cv2.warpPerspective(image_to_align, M, (width, height))
        cv2.imwrite("aligned.png", aligned)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), max_features))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(image_to_align, kp1, reference_image, kp2, good, None, **draw_params)
    img3 = cv2.resize(img3, None, fx=0.3, fy=0.3)
    cv2.imwrite("matches.png", img3)

def main():
    reference_image = cv2.imread("reference_img.png")
    harris(reference_image.copy())

    image_to_align = cv2.imread("align_this.jpg")
    alignment_sift(image_to_align, reference_image, 10, 0.7)

if __name__ == "__main__":
    main()