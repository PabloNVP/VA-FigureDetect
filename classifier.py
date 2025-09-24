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

def save_dataset(X : list, Y : list) -> None:
  with open('data.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    for x_row, y_val in zip(X, Y):
        writer.writerow(x_row + [y_val])

class Classifier:
  def __init__(self):
    if os.name == "nt":
      self.webcam = cv.VideoCapture(0, cv.CAP_DSHOW) # WINDOWS
    elif os.name == "posix":
      self.webcam = cv.VideoCapture(2) # LINUX / OS 

    self.C = []
    self.X = []
    self.Y = []

    self.create_menu()
    self.load_contours()

  def create_menu(self) -> None:
    cv.namedWindow("Manual")
    cv.createTrackbar("Umbral", "Manual", 140, 255, lambda x: None)
    cv.createTrackbar("Morfologico", "Manual", 2, 20, lambda x: None)
    cv.createTrackbar("Area_minima", "Manual", 800, 5000, lambda x: None)
    cv.createTrackbar("Area_maxima", "Manual", 100000, 600000, lambda x: None)
    cv.createTrackbar("Umbral_Match", "Manual", 1, 100, lambda x: None)

  def load_contours(self):
    for fig in FIGURE:
      th_manual = self.threshold_manual(fig)
      frame_morfo = self.morphological_operations(th_manual)
      contornos = self.find_contours(frame_morfo)
      if contornos:
        self.C.append(contornos[0])

  def threshold_manual(self, frame):
    t = cv.getTrackbarPos("Umbral", "Manual")
    _, th_manual = cv.threshold(frame, t, 255, cv.THRESH_BINARY_INV)
    return th_manual

  def threshold_otsu(self, frame):
    _, th_otsu = cv.threshold(frame, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return th_otsu

  def threshold_triangle(self, frame):
    _, th_triangle = cv.threshold(frame, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_TRIANGLE)
    return th_triangle
  
  def morphological_operations(self, frame):
    ksize = cv.getTrackbarPos("Morfologico", "Manual")
    if ksize < 1: ksize = 1
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))

    op_clean = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
    op_clean = cv.morphologyEx(op_clean, cv.MORPH_CLOSE, kernel)

    return op_clean
  
  def find_contours(self, frame):
    contours, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    min_area = cv.getTrackbarPos("Area_minima", "Manual")
    max_area = cv.getTrackbarPos("Area_maxima", "Manual")
    contours_filtered = [
      cnt for cnt in contours if (min_area <= cv.contourArea(cnt) <= max_area)
    ]
    return contours_filtered
  
  def process_contours(self, frame, contours):
    threshold_match = cv.getTrackbarPos("Umbral_Match", "Manual") / 100
    figures = []

    for cnt in contours:
      x, y, w, h = cv.boundingRect(cnt)
    
      best : str = None
      min_distance : float = float('inf')

      for index, contorno_ref in enumerate(self.C):
        distance = cv.matchShapes(cnt, contorno_ref, cv.CONTOURS_MATCH_I1, 0.0)
        if distance < min_distance:
          min_distance = distance
          best = FIGURE_TAG[index]

      figures.append((cnt, best if min_distance <= threshold_match else "Desconocido", x, y, w, h))

    return figures

  def draw_figures(self, frame, figures):
    for _, tag, x, y, w, h in figures:
      colour = (0,255,0) if tag != "Desconocido" else (0,0,255)
      cv.putText(frame, f"{tag}", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
      cv.rectangle(frame, (x, y), (x+w, y+h), colour, 2)

  def run(self):
    while True:
      ret, frame = self.webcam.read()

      if not ret: break

      gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

      frame_thresold = self.threshold_manual(gray)

      frame_morfo = self.morphological_operations(frame_thresold)

      contours = self.find_contours(frame_morfo)

      figures = self.process_contours(frame, contours)

      self.draw_figures(frame, figures)
      cv.putText(frame, f"Figuras detectadas: {len(contours)}", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)
      cv.imshow("Debug", frame_morfo)
      cv.imshow("Manual", frame)
  
      key = cv.waitKey(30) & 0xFF

      if key == ord('q'): 
        break

      if key == ord('c') and figures:
        for cnt, label, _, _, _, _ in figures:
          if label != "Desconocido":
            self.X.append(cv.HuMoments(cv.moments(cnt)).flatten().tolist())
            self.Y.append(label)
            print(f"Guardado contorno de referencia: {label}")
      
      if key == ord('g'):
        save_dataset(self.X, self.Y)

  def __del__(self):
    self.webcam.release()
      
cv.destroyAllWindows()
    
if __name__ == "__main__":
  csf = Classifier()
  csf.run()
