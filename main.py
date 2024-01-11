import cv2
import numpy as np

# Get the video input from the first Video-Input Device
# On Raspberry PI this to be set to `1` as depth camera will be present at `0` 
cap=cv2.VideoCapture(0)

# Defining the HSV range for blue color
lower_range=np.array([100,150,0])
upper_range=np.array([140,255,255])

# Defining the HSV range for green color
lower_range_G=np.array([37,87,0])
upper_range_G=np.array([69,255,255])

while True:
    #Getting HSV-converted video frame 
    ret,frame=cap.read()
    X_resize, Y_resize = 640,480
    frame=cv2.resize(frame,(X_resize,Y_resize))
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    ### Creating mask for Blue color
    mask=cv2.inRange(hsv,lower_range,upper_range)
    T_mask = mask
    # Generating Binary Threshold (BT) for the mask
    _,mask1=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    # Generating Contours from the BT
    cnts,_=cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        x=600
        if cv2.contourArea(c)>x:
            #DEBUG-ONLY: Generates a outline around the object
            x,y,w,h=cv2.boundingRect(c)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            #-----------------------------------------------

            # Calculate the centroid of the Object located
            M = cv2.moments(c)
            if M["m00"] != 0:
                target_x = int(M["m10"] / M["m00"])
                target_y = int(M["m01"] / M["m00"])

                cv2.putText(frame,(f"Container found at: {target_x}, {target_y} "),(10,60),cv2.FONT_ITALIC,0.6,(0,0,255),2)
                cv2.putText(frame,("O"),(target_x,target_y),cv2.FONT_ITALIC,0.6,(0,0,255),2)
                if(abs(X_resize/2-target_x)>50):
                    # In case we need to add rover stabilization
                    pass
                if(target_x>X_resize/2):
                    cv2.putText(frame,(f"Move right"),(10,100),cv2.FONT_ITALIC,0.6,(0,0,255),2)
                    # Send TURN_RIGHT enum to slave to turn rover right
                else:
                    cv2.putText(frame,(f"Move left"),(10,100),cv2.FONT_ITALIC,0.6,(0,0,255),2)
                    # Send TURN_LETT enum to slave to turn rover left

    ### Creating mask for Green color
    mask=cv2.inRange(hsv,lower_range_G,upper_range_G)
    T_mask = T_mask | mask
    _,mask1=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    cnts,_=cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        x=600
        if cv2.contourArea(c)>x:
            #DEBUG-ONLY: Generates a outline around the object
            x,y,w,h=cv2.boundingRect(c)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            #-----------------------------------------------

            # Calculate the centroid of the Object located
            M = cv2.moments(c)
            if M["m00"] != 0:
                target_x = int(M["m10"] / M["m00"])
                target_y = int(M["m01"] / M["m00"])

                cv2.putText(frame,(f"Container found at: {target_x}, {target_y} "),(320,60),cv2.FONT_ITALIC,0.6,(0,0,255),2)
                cv2.putText(frame,("O"),(target_x,target_y),cv2.FONT_ITALIC,0.6,(0,0,255),2)
                if(abs(X_resize/2-target_x)>50):
                    # In case we need to add rover stabilization
                    pass
                if(target_x>X_resize/2):
                    cv2.putText(frame,(f"Move right"),(320,100),cv2.FONT_ITALIC,0.6,(0,0,255),2)
                    # Send TURN_RIGHT enum to slave to turn rover right
                else:
                    cv2.putText(frame,(f"Move left"),(320,100),cv2.FONT_ITALIC,0.6,(0,0,255),2)
                    # Send TURN_LETT enum to slave to turn rover left
    
    cv2.imshow("Debug Frame",frame)
    cv2.imshow("Object Mask",T_mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()