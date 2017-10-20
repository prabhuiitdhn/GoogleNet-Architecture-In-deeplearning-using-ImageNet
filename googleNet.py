#for execution
#python googleNet.py --image images/jemma.png --prototxt bvlc_googlenet.prototxt --model bvlc_googlenet.caffemodel --labels synset_words.txt


#Importing the neccessary package for ImageNetclassification
#Numpy is used for numerical expression
#argparse is used for argument passing in the model/architecture
#time package have time-related function which coudl be used to find tyhe starting time and ending etc..
#CV2 is OpenCV package 
import numpy as np
import argparse
import time
import cv2

#passing the argument
ap=argparse.ArgumentParser()
#pass the path of image file which has to be recognised 
ap.add_argument("-i","--image", required=True,help="Path to the input image")

#pass the path of configuration file of caffe model which would be in .text file having the full configuartion of model
ap.add_argument("-p","--prototxt", required=True, help="path to the 'caffe' deploy prototxt file")

#pass the path of pre-trained Model which would be used for classfication \
#which will have set conv, pooling, Relu layer and full connected layer
#gere we're using GoogleNet model of CNN  
ap.add_argument("-m","--model", required=True, help="path to the pre-trained model")

#passing the path of class label will have unique identification of class with thier values
ap.add_argument("-l","--labels", required=True, help="Path to the ImageNet labesl")
args=vars(ap.parse_args())

#Reading the image
image=cv2.imread(args["image"])

#splitting the ending and begining whitespace for create an unique identification  
rows=open(args["labels"]).read().strip().split("\n")

#read all classes in label file
classes=[r[r.find(" ")+1:].split(",")[0] for r in rows]

#blobFromImage return 4-dimension image which resizes and crop on the center of image then it perfrom mean subtraction for normalisation
#will resize of 224X224 pixels while performing mean subtraction
#(104, 117, 123) to normalise and after this execution will have the shape of
#(1, 3, 224, 224)
blob=cv2.dnn.blobFromImage(image, 1, (224, 224),(104, 117, 123))


#loading the model from disk using dnn module of openCV
print("[Info] loading model...")
net=cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
net.setInput(blob)
start=time.time()

#used to network forward propogation after executing the blob image
preds=net.forward()
end=time.time()

#printing the classification occured and how much time has been taken to perform that
print("[Info] classification took {:.5} seconds".format(end-start))

#performing the operation of predcition probablity in sorting form 
#and take top 5 predictions of it
idxs=np.argsort(preds[0])[::-1][:5]

#loop over the top 5 predictions and display them
for(i, idx) in enumerate(idxs):
	if i==0:
		text="Label: {}, {:.2f}%".format(classes[idx], 
		       preds[0][idx]*100)
		cv2.putText(image, text, (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)
		
        #display predicted labels and + associated probablity to the console
		print("[Info] {}.label: {}, probablity: {:.5}".format(i+1, 
			classes[idx], preds[0][idx]))

cv2.imshow("Image", image)
cv2.waitKey(0)



