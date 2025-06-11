# from ultralytics import YOLO
# import cv2

# model = YOLO(r"C:\Users\mayan\Downloads\trained\yolo11_best .pt")
# results = model.predict(source=0, show=True, conf=0.15)

from ultralytics import YOLO
import cv2
import time

# Load the trained model
model = YOLO(r"C:\Users\mayan\Downloads\trained\yolo11_best .pt")

# Define your buzzer trigger function
def buzzer():
    print("🚨 Buzzer ON: No helmet detected!")
    # Example for Raspberry Pi:
    # GPIO.output(BUZZER_PIN, GPIO.HIGH)
    # time.sleep(1)
    # GPIO.output(BUZZER_PIN, GPIO.LOW)

# Start detection loop
cap = cv2.VideoCapture(0)

# video_path = r"C:\Users\mayan\Downloads\clip3.mp4"  # ← CHANGE THIS PATH
# cap = cv2.VideoCapture(video_path)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO prediction on the frame
    results = model.predict(source=frame, conf=0.5, save=False, verbose=False)

    # Parse detection results
    detections = results[0].boxes
    classes = results[0].names

    helmet_found = False
    no_helmet_found = False

    for box in detections:
        cls_id = int(box.cls[0])  # class index
        label = classes[cls_id].lower()

        # Draw bounding box
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 255), 2)
        cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0 , 255), 2)

        if "hat" in label:
            helmet_found = True
        elif "person" in label or "hat" in label:
            no_helmet_found = True

    # Trigger buzzer if person without helmet detected
    if no_helmet_found and not helmet_found:
        buzzer()

    cv2.imshow("Helmet Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
