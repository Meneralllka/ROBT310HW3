import cv2

cap = cv2.VideoCapture("dataset.mp4")
cnt = 0

while 1:
    ret, frame = cap.read()
    if ret:
        if cnt % 100 == 0:
            #cv2.imshow("window", frame)
            cv2.imwrite("dataset/" + str(cnt) + ".jpg", frame)
            print(cnt, "saved")
            if cv2.waitKey(1) == "q":
                break
        cnt += 1
    else:
        break
cv2.destroyAllWindows()
cap.release()
