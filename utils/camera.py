import cv2

class Camera:
    def __init__(self, source=0):
        # Initialize the video source (default is 0 for webcam)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise Exception("Could not open video source")

    def __del__(self):
        # Release the video source when the object is destroyed
        if self.cap.isOpened():
            self.cap.release()

    def next(self):
        # Read the next frame from the video source
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            print("Could not read frame from video source")
            return None


if __name__ == "__main__":
    # Usage example
    cam = Camera()  # Use 0 for webcam, or 'http://ip-address:port' for an IP camera
    while True:
        frame = cam.next()
        print(frame.shape)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()