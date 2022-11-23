import cv2
from enum import Enum

from os import path

DATASET_ROOT = r"C:\Users\J-Dau\Projekte\Datasets\tum_deformable-3d-reconstruction"


class DatasetModel(Enum):
    DUCK = "Duck"
    SNOOPY = "Snoopy"


MODEL = DatasetModel.DUCK.value


def ORB_matching(imgs, nFeatures=100, masked_imgs=None):

    if masked_imgs:
        img1, img2 = masked_imgs
    else:
        img1, img2 = imgs

    orb = cv2.ORB_create(nfeatures=nFeatures)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    return cv2.drawMatches(imgs[0], kp1, imgs[1], kp2, matches[:50], None)


def FLANN_matching(imgs, nFeatures=100, masked_imgs=None, detector="ORB"):

    search_params = dict(checks=50)   # or pass empty dictionary

    if detector == "ORB":
        detector = cv2.ORB_create(nfeatures=nFeatures)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,     # 20
                            multi_probe_level=1)  # 2
    elif detector == "SIFT":
        detector = cv2.SIFT_create(nfeatures=nFeatures)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    else:
        print(f"Error: Unsupported detector '{detector}")
        return

    if masked_imgs:
        img1, img2 = masked_imgs
    else:
        img1, img2 = imgs

    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)
    return cv2.drawMatchesKnn(imgs[0], kp1, imgs[1], kp2, matches, None, **draw_params)


def main():
    first_index = 170
    second_index = 200
    nFeatures = 50

    dataset_path = path.join(DATASET_ROOT, MODEL)

    img1 = cv2.imread(path.join(dataset_path, 'color_{:06d}.png'.format(first_index)))
    img2 = cv2.imread(path.join(dataset_path, 'color_{:06d}.png'.format(second_index)))

    mask1 = cv2.imread(path.join(dataset_path, 'omask_{:06d}.png'.format(first_index)))
    mask2 = cv2.imread(path.join(dataset_path, 'omask_{:06d}.png'.format(second_index)))

    masked_img1 = cv2.bitwise_and(img1, mask1)
    masked_img2 = cv2.bitwise_and(img2, mask2)

    # match_img = ORB_matching((img1, img2), nFeatures, masked_imgs=(masked_img1, masked_img2))
    match_img = FLANN_matching((img1, img2), nFeatures, masked_imgs=(masked_img1, masked_img2), detector="")


    cv2.imshow("", match_img)
    cv2.waitKey()


if __name__ == "__main__":
    main()
