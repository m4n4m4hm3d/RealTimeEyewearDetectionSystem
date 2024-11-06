from ultralytics import YOLO
import cv2 as cv

model = YOLO('models/best.pt')

img = cv.imread("Images/face.jpeg") #Trained on custom dataset

results = model(img)

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    results = model(frame)
    
    if len(results[0].boxes)>0:
        x1, y1, x2, y2 = results[0].boxes.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cls = results[0].boxes.cls.tolist()
        
        if cls == 0:
            cv.putText(frame, "Glasses", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv.putText(frame, "No Glasses", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
             
    cv.imshow("frame", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
    
