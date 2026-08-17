import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

drawingUtils = mp.solutions.drawing_utils
drawingStyles = mp.solutions.drawing_styles
faceMesh = mp.solutions.face_mesh

detector = faceMesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6)

while cap.isOpened():

  success, frame = cap.read()
  if not success: continue

  frame = cv2.flip(frame, 1)
  RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  result = detector.process(RGBframe)

  if result.multi_face_landmarks:
    for face_landmark in result.multi_face_landmarks:

      nose_tip = face_landmark.landmark[0]
      h, w, _ = frame.shape

      pixel_x = int(nose_tip.x * w)
      pixel_y = int(nose_tip.y * h)

      cv2.circle(frame, (pixel_x, pixel_y), radius=3, color=(0, 255, 0), thickness=-1)
      drawingUtils.draw_landmarks(image=frame, 
                                  landmark_list=face_landmark, 
                                  connections=faceMesh.FACEMESH_TESSELATION, 
                                  landmark_drawing_spec=None, 
                                  connection_drawing_spec=drawingStyles.get_default_face_mesh_tesselation_style())
      
  cv2.imshow("My Face", frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()