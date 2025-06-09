from ultralytics import YOLO
import cv2

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Define path to video file
#source = "D:/landing in  England.mp4"

cap=cv2.VideoCapture("D:/landing in  England.mp4")
while cap.isOpened():
    success,frame=cap.read()
    if not success:
        break
    results=model(frame)
    annotated_frame=results[0].plot()
    cv2.imshow("YOLOv11 detection",annotated_frame)
    if cv2.waitKey(1)&0xFF==ord("q"): #按'q'退出
        break
cap.release()
cv2.destroyAllWindows()
