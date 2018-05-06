import sys
# Error in print Using Tensorflow Backend NOQA
# https://github.com/keras-team/keras/issues/1406 NOQA
stderr = sys.stderr  # NOQA
sys.stderr = open('/dev/null', 'w')  # NOQA
import kivy
kivy.require('1.9.1')
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.widget import Widget
import cv2
import os
import numpy as np
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import image_to_string
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
import sqlite3
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2

# fix 3.6 tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf  # NOQA
sys.stderr = stderr  # NOQA



def test_img(frame):
    image = cv2.resize(frame, (128, 128))
    img = image.astype("float") / 255.0
    img = img_to_array(img)
    return np.expand_dims(img, axis=0)


def detection_image(img):
    frame = test_img(img)
    model = load_model('positive_negative.model')
    negative, positive = model.predict(frame)[0]
    if negative > 0.89:
        frame = cv2.pyrDown(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        _, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros(bw.shape, dtype=np.uint8)
        for idx in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[idx])
            if h < 90 and w < 400:
                mask[y:y+h, x:x+w] = 0
                cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
                r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
                if r > 0.45 and w > 7 and h > 7:
                    cv2.rectangle(frame, (x, y), (x+w-1, y+h-1), (255, 255, 255), 2)
                    sub_image = frame[y:y+h, x:x+w]
                    img_file = str(y) + ".jpg"
                    text = get_string(sub_image)
                    return add_widget(text=text, frame=frame)
    return ''


def get_string(img_path):
    # Read image with opencv
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    result = pytesseract.image_to_string(Image.open(img))
    return result
       

class KivyCamera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        lbl = Label(text=kwargs.get('text', ''))
        lbl.font_size= '60sp'
        lbl.bold = True
        lbl.color=[0, 0, 1, 0.5]
        self.add_widget(lbl)        
        self.orientation = 'vertical'        
        Clock.schedule_interval(self.update, 1.0 / fps)


    def add_widget(self, lbl):
        conn = sqlite3.connect("test.db")
        cur = conn.cursor()
        cur.execute("INSERT INTO Valores(text) values(?);", (lbl.text,))
        cur.close()
        conn.close()        
        super(KivyCamera, self).add_widget(lbl)         

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            text = detection_image(buf1)
            #image_texture = WidgetRaiz(text=text)
            self.texture = image_texture


class CamApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(-1)
        self.my_camera = KivyCamera(capture=self.capture, fps=30)
        return self.my_camera

    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.capture.release()


if __name__ == '__main__':
    CamApp().run()
