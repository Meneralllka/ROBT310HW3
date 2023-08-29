import cv2
import numpy as np
import os
import time
import math

size = 5


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
        #print(B.dtype, B)
        if not isinstance(overall_comp, np.intc):
            #print(overall)
            overall_comp = overall
            print(type(overall_comp))
            ideal = img_list[i]
        else:
            if overall_comp > overall:
                print(overall)
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
    img_list = get_images("src")
    base_img = cv2.imread("src/base.jpg")
    cv2.imshow("window", base_img)
    print(type(base_img))
    for x in range(base_img.shape[0] // size):
        for y in range(base_img.shape[1] // size):
            base_cut = base_img[x * size: (x + 1) * size,
                              y * size: (y + 1) * size]
            ideal = closest_image(img_list, base_cut)
            base_img[x * size: (x + 1) * size,
            y * size: (y + 1) * size] = ideal
            cv2.imshow("cut", base_img)
            cv2.waitKey(1)
            #print(x, y)
            #time.sleep(0.01)

    print("done")
    cv2.imshow("window", base_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
