import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import tensorflow as tf



def find_contours(dimensions, img) :

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sorting largest 15 contours contours according size for license plate
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
     # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    x_cntr_list = []
    img_res = []
    
    for cntr in cntrs :
        #detects contour in binary image and returns the coordinates of rectangle enclosing it
        x, y, w, h = cv2.boundingRect(cntr)
        
        #checking the dimensions of the contour to filter out the characters by contour's size
        if w > lower_width and w < upper_width and h > lower_height and h < upper_height :
            x_cntr_list.append(x) #stores the x coordinate of the character's contour, to used later for indexing the contours
            
            # Create image using numpy 24*44
            char_copy = np.zeros((44,24))
            
            # Extracting each character using the enclosing rectangle's coordinates.
            char = img[y:y+h, x:x+w]
            
            char = cv2.resize(char, (20, 40))
          
            # Show image plate  
            cv2.rectangle(img, (x,y), (w+x, y+h), (50,21,200), 2)
            plt.imshow(img)

            
            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0
            
            #List that stores the character's binary image
            img_res.append(char)
            
    plt.show()
    #arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        # stores character images according to their index
        img_res_copy.append(img_res[idx])
    img_res = np.array(img_res_copy)

    return img_res


def segment_characters(image) :

    # Preprocess cropped license plate image using Otsu's Binarization thresholding
    _, img_binary_lp = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Morphological Dilation and Erosion to remove the noise
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))
    
    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]
    
    
    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6, LP_WIDTH/2, LP_HEIGHT/8, 2*LP_HEIGHT/3]
    plt.imshow(img_binary_lp)
    plt.show()

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list

# CNN loading model Traning to extract the right number
model = tf.keras.models.load_model('F:/Work/Poject/Training_CNN' )

# Capture Live Video from Camera.
cap = cv2.VideoCapture(0)

# Set the width and height for the frame.
cap.set(3,900)
cap.set(4,900)

# Check if camera opened successfully.
if (cap.isOpened()== False):
  print("Error opening Camera.")

while True:
    # Capture frame-by-frame.
    success,img = cap.read()
    ##############################
    
    # Show date_time in frame during live capture  
    date_time = str(datetime.now().strftime("Date %d-%m-%Y Time %I:%M:%S:%f"))
    cv2.putText(img, date_time, (10, 80), cv2.FONT_ITALIC, 1, (0, 105, 255), thickness=2)
    ################################# 
    
    # Convert Color Frame to Gray-Scale.
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Load Haar-Cascade file for plate-number detection.
    PlateCascade = cv2.CascadeClassifier("haarcascade_plate_number.xml")
    # Apply the plate-number detection on gray-image.
    numberPlates = PlateCascade.detectMultiScale(gray_image, scaleFactor= 1.6, minNeighbors= 7) 
    
    # Draw Rectangle around the detected plate 
    for (x, y, w, h) in numberPlates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2 ) 
        # Crop the detected plate from original frame.          
        imgRoi = img[y:y+h, x:x+w]
        # Convert detected Frame to Gray-Scale.
        gray_Roi = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2GRAY)
        plt.imshow(imgRoi)
        # Display the Gray cropped plate
        cv2.imshow("ROI", gray_Roi)
 
        char = segment_characters(gray_Roi)

        # return 
        def fix_dimension(img): 
            new_img = np.zeros((28,28,3))
            for i in range(3):
                new_img[:,:,i] = img
            return new_img
  
        dic = {}
        characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i,c in enumerate(characters):
            dic[i] = c
 
        output = []
        for i,ch in enumerate(char): 
            img_ = cv2.resize(ch, (28,28))
            img = fix_dimension(img_)
            #preparing image for the model
            img = img.reshape(1,28,28,3) 
            
            y = model.predict(img)
            # Return the highest expected probability of character
            character = dic[np.argmax(y[0])]
            # Storing the result in a list
            output.append(character) 
            
        # Convert list to string    
        plate_number = ''.join(output)
        print('plate_number',plate_number)
        
        #return present time
        now_time = datetime.now().strftime("%I:%M:%S:%f")
        print('time : ',now_time)  
        
        cv2.waitKey(400)
        
    # Display the resulting frame    
    try:  
        cv2.imshow("video",img)
    except:
        pass
    
    # Press Escape-ESC- on keyboard to stop recording
    key = cv2.waitKey(1)
    if key == 27:
        break
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()    

