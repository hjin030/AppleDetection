import cv2
import numpy as np

#Load yolo
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# print(classes)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

#load images and operation with images
img = cv2.imread("apple/10.jpg")
# img = cv2.resize(img,None,fx=0.4,fy=0.4)
height,width,channels = img.shape

#Detecting objects
# blob = cv2.dnn.blobFromImage(img,0.00392, (416,416), (0,0,0),True,crop=False)
blob = cv2.dnn.blobFromImage(img,0.008, (416,416), (0,0,0),True,crop=False)


# for b in blob:
#     for n, img_blob in enumerate(b):
#         cv2.imshow(str(n),img_blob)

net.setInput(blob)
outs = net.forward(output_layers)

# print(outs)


#showing information on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[30:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # cv2.circle(img, (center_x, center_y), 10, (0,255,0),2)
            print("apple is located at:x:",w,"and y :",h)
            # print(h)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)


number_objects_detected = len(boxes)
indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    x,y,w,h = boxes[i]
    label = str(classes[class_ids[i]])
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(img,label,(x,y+30),font,1,(0,0,0),3)



print("Number of apple detected are:",len(boxes))
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()





