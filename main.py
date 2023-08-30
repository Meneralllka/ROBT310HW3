'''
1 (65%) - main task accomplishment -- DONE
2 (5%) - for development of personal image database for creating the mosaic -- DONE
3 (5%) - for implementation of a different shape (circular, parallelogram, etc…)
4 (5%) - awarded if the size of a patch is adjustable -- DONE
5 (10%) For questions 1‐4, create a document consisting of your inputs, outputs, your observations,
problems you have faced, solutions indicating how you have overcome the problems, and other points
that you think are necessary.
6 (10%) For questions 1‐5, create a video describing your solutions.
'''


import cv2
import numpy as np
import os

global size
size = 0


def nothing(x):
    pass


def closest_image(img_list, base_cut):
    ideal = None
    overall_comp = None
    for i in range(len(img_list)):
        img_list[i] = img_list[i].astype('int32')
        base_cut = base_cut.astype('int32')
        diff = np.subtract(img_list[i], base_cut)
        diff = np.absolute(diff)
        img_list[i] = img_list[i].astype('uint8')
        (B, G, R) = cv2.split(diff)
        B = np.sum(np.square(B))
        G = np.sum(np.square(G))
        R = np.sum(np.square(R))
        overall = B + G + R
        if not isinstance(overall_comp, np.intc):
            overall_comp = overall
            ideal = img_list[i]
        else:
            if overall_comp > overall:
                overall_comp = overall
                ideal = img_list[i]
    return ideal


def get_images(folder):
    img_name_list = os.listdir(folder)
    img_list = []
    for name in img_name_list:
        full_name = os.path.join(folder, name)
        img = cv2.imread(full_name)
        img = cv2.resize(img, (size, size))
        img_list.append(img)
    return img_list


def run():
    cv2.namedWindow("image")
    cv2.createTrackbar('Patch size', 'image', 5, 50, nothing)
    cv2.createTrackbar('Start', 'image', 0, 1, nothing)

    base_img = cv2.imread("base.jpeg")
    edited = base_img.copy()

    finished = 0
    while 1:
        res = cv2.hconcat(base_img, edited)
        cv2.imshow("image", edited)
        global size
        size = cv2.getTrackbarPos('Patch size', 'image')
        start = cv2.getTrackbarPos('Start', 'image')
        if start and not finished:
            print(size)
            img_list = get_images("dataset")
            for x in range(base_img.shape[0] // size):
                if finished:
                    break
                for y in range(base_img.shape[1] // size):
                    start = cv2.getTrackbarPos('Start', 'image')
                    if not start:
                        finished = 1
                        edited = base_img.copy()
                    if finished:
                        break
                    base_cut = base_img[x * size: (x + 1) * size,
                               y * size: (y + 1) * size]
                    ideal = closest_image(img_list, base_cut)
                    edited[x * size: (x + 1) * size,
                    y * size: (y + 1) * size] = ideal
                    cv2.imshow("image", edited)
                    if cv2.waitKey(1) == "q":
                        break
                    print(x, y)
            finished = 1
        elif not start and finished:
            finished = 0

        if cv2.waitKey(1) == "q":
            break



    print("done")
    cv2.imshow("window", base_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
