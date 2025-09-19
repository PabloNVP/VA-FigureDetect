import cv2 as cv
import os
import csv

SIZE_IMG = (256, 256)

FIGURE = [
  cv.resize(cv.imread('./assets/circle.png', cv.IMREAD_GRAYSCALE), SIZE_IMG),
  cv.resize(cv.imread('./assets/square.png', cv.IMREAD_GRAYSCALE), SIZE_IMG),
  cv.resize(cv.imread('./assets/triangle.png', cv.IMREAD_GRAYSCALE), SIZE_IMG)
]

FIGURE_TAG = [
  "Circulo", "Cuadrado", "Triangulo"
]

def save_dataset(X, Y):
  with open('data.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    for x_row, y_val in zip(X, Y):
        writer.writerow(x_row + [y_val])

class Classifier:
  def __init__(self):
    if os.name == "nt":
      self.webcam = cv.VideoCapture(0, cv.CAP_DSHOW) # WINDOWS
    elif os.name == "posix":
      self.webcam = cv.VideoCapture(0) # LINUX / OS

    self.C = self.X = self.Y = []

    self.create_menu()
    self.load_contours()

  def create_menu() -> None:
    cv.namedWindow("Manual")
    cv.createTrackbar("Umbral", "Manual", 40, 255, lambda x: None)
    cv.createTrackbar("Morfologico", "Manual", 3, 20, lambda x: None)
    cv.createTrackbar("Area", "Manual", 300, 5000, lambda x: None)
    cv.createTrackbar("Umbral_Match", "Manual", 0, 100, lambda x: None)

  def load_contours(self):
    for figure in FIGURE:
      th_manual = self.threshold_manual(figure)
      frame_morfo = self.morphological_operations(th_manual)
      cnt = self.find_contours(frame_morfo)[0]
      self.C.append(cnt)

  def threshold_manual(frame):
    t = cv.getTrackbarPos("Umbral", "Manual")
    _, th_manual = cv.threshold(frame, t, 255, cv.THRESH_BINARY_INV)
    return th_manual

  def threshold_otsu(frame):
    _, th_otsu = cv.threshold(frame, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return th_otsu

  def threshold_triangle(frame):
    _, th_triangle = cv.threshold(frame, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_TRIANGLE)
    return th_triangle
  
  def morphological_operations(frame):
    ksize = cv.getTrackbarPos("Morfologico", "Manual")
    if ksize < 1: ksize = 1
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))

    op_clean = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
    op_clean = cv.morphologyEx(op_clean, cv.MORPH_CLOSE, kernel)

    return op_clean
  
  def find_contours(frame):
    contours, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    min_area = cv.getTrackbarPos("Area", "Manual")
    contours_filtered = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]
    return contours_filtered
  
  def process_contours(self, frame, contours):
    for cnt in contours:
      #cv.drawContours(frame, [contorno], -1, (0,255,0), 2)
  
      x, y, w, h = cv.boundingRect(cnt)

      threshold_match = cv.getTrackbarPos("Umbral_Match", "Manual") / 100

      best : str = None
      min_distance : float = float('inf')
      for index, contorno_ref in enumerate(self.C):
        distance = cv.matchShapes(cnt, contorno_ref, cv.CONTOURS_MATCH_I1, 0.0)
        if distance < min_distance:
            min_distance = distance
            best = FIGURE_TAG[index]

      if min_distance < threshold_match:
        cv.putText(frame, f"{best}", (x, y-10),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
      else:
        cv.putText(frame, "Desconocido", (x, y-10),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

  def run(self):
    while True:
      ret, frame = self.webcam.read()

      if not ret: break

      gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

      frame_thresold = self.threshold_manual(gray)

      frame_morfo = self.morphological_operations(frame_thresold)

      contours = self.find_contours(frame_morfo)

      self.process_contours(frame, contours)

      cv.imshow("Debug", frame_morfo)
      cv.imshow("Manual", frame)
  
      key = cv.waitKey(30) & 0xFF

      if key == ord('q'): break

      if key == ord('c') and contours:
        nombre = input("Ingrese etiqueta de esta forma: ")
        for index, cnt in enumerate(contours):
          self.C.append(cnt.copy())
          self.X.append(cv.HuMoments(cv.moments(cnt)).flatten().tolist())
          self.Y.append(nombre)
          print(f"Guardado contorno de referencia: {nombre}")


    def __del__(self):
      self.webcam.release()
      
cv.destroyAllWindows()
    
