import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

faceMesh = mp.solutions.face_mesh
detector = faceMesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6)

while cap.isOpened():

  success, frame = cap.read()
  if not success: continue

  frame = cv2.flip(frame, 1)
  RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  result = detector.process(RGBframe)

  cv2.imshow("My Face", frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()