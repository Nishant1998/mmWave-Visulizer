import cv2

# Replace the URL with your actual stream URL.
stream_url = "http://user:pass@172.17.73.199:8080/video"
cap = cv2.VideoCapture(stream_url)
print(cap.isOpened())
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('IP Camera stream',frame)

    # Exit if Q key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
