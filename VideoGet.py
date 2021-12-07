from threading import Thread
import cv2
import time

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        self.stopped = False
        self.grabbed = True
        print("getter is restarted")
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            print(self.stopped)
            if not self.grabbed:
                print("grabbed is false")
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
            #time.sleep(0.066)

    def stop(self):
        self.stopped = True
        print("stopped is true")