import cv2
import numpy as np
import os
import time

# def st_dev(img):

def get_images(folder):
    img_name_list = os.listdir(folder)
    img_list = []
    for name in img_name_list:
        full_name = os.path.join(folder, name)
        img_list.append(cv2.imread(full_name))
    return img_list
def run():
    img_list = get_images("src")
    if (len(img_list)):
        for img in img_list:
            cv2.imshow("window", img)
            time.sleep(0.1)
            if cv2.waitKey(1) == "q":
                break
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
