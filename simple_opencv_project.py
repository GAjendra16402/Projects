#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2


# In[ ]:


import cv2

def main():
    # Open a connection to the webcam (0 represents the default camera)
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame is read successfully
        if not ret:
            print("Error: Couldn't read frame. Exiting...")
            break

        # Display the frame in a window
        cv2.imshow("Webcam Feed", frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# In[ ]:


q


# In[ ]:




