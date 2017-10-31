# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 00:18:45 2017

@author: anura
"""

import cv2
import numpy as np
import imutils
import argparse
from imutils.perspective import four_point_transform,order_points
from skimage.filters import threshold_adaptive
from skimage.segmentation import clear_border
from keras.models import load_model
from sudoku import SolveSudoku

#parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m","--model",required=True,help="Path to trained model...")
args = vars(ap.parse_args())

#capture from the webcam
cap = cv2.VideoCapture(0)
poly = None         #to capture the sudoku co-ordinates

while True:
    ret,image = cap.read()
    #resize image
    image = imutils.resize(image,width=800)
    cv2.imshow("Original",image)
    #to gray
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #blur the image
    blurred = cv2.GaussianBlur(gray,(5,5),0)
    #apply adaptive thresholding
    thresh = threshold_adaptive(blurred,block_size=5,offset=1).astype("uint8")*255
    cv2.imshow("Thresholded",thresh)

    key = cv2.waitKey(1) & 0XFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        #find countours
        cnts,_ = cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        #sort the contours with highest area first
        cnts = sorted(cnts,key=cv2.contourArea,reverse=True)
        mask = np.zeros(thresh.shape,dtype="uint8")
        c = cnts[1]
        clone = image.copy()

        #approximate the contours
        peri = cv2.arcLength(c,closed=True)
        poly = cv2.approxPolyDP(c,epsilon=0.02*peri,closed=True)

        #only if a valid region
        if len(poly) == 4:
            cv2.drawContours(clone,[poly],-1,(0,0,255),2)
            #apply perspective transform
            warped = four_point_transform(image,poly.reshape(-1,2))
            cv2.imshow("Contours",clone)
            cv2.imshow("Warped",warped)
            break

key = cv2.waitKey(0)

#just in case if the captured one is not the actual puzzle
if key&0XFF == ord("q"):
    exit()

#convert to gray and find the sliding window width & height
warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
winX = int(warped.shape[1]/9.0)
winY = int(warped.shape[0]/9.0)

#load the trained model
model = load_model(args["model"])

#empty lists to capture recognized digits and center co-ordinates of the cells
labels = []
centers = []

#slide the window through the puzzle
for y in xrange(0,warped.shape[0],winY):
    for x in xrange(0,warped.shape[1],winX):
        #slice the cell
        window = warped[y:y+winY,x:x+winX]
        #sanity check
        if window.shape[0] != winY or window.shape[1] != winX:
            continue

        clone = warped.copy()
        digit = cv2.resize(window,(28,28))
        _,digit = cv2.threshold(digit,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        #clear borders
        digit = clear_border(digit)
        cv2.imshow("Digit",digit)

        #whether an empty cell or not
        numPixels = cv2.countNonZero(digit)
        if numPixels<5:
            label = 0
        else:
            label = model.predict_classes([digit.reshape(1,28,28,1)])[0]
        labels.append(label)
        centers.append(((x+x+winX)//2,(y+y+winY+6)//2))
        #draw rectangle for each cell
        cv2.rectangle(clone,(x,y),(x+winX,y+winY),(0,0,255),2)
        cv2.imshow("Window",clone)
        cv2.waitKey(0)

#convert to numpy array of 9x9
grid = np.array(labels).reshape(9,9)

#find the indices of empty cells
gz_indices = zip(*np.where(grid==0))

#center co-ordinates of all the cells
gz_centers = np.array(centers).reshape(9,9,2)

#solve the sudoku
sudoku = SolveSudoku(grid)
grid = sudoku.solve()

#fill the solved numbers in empty cells
for row,col in gz_indices:
    cv2.putText(warped,str(grid[row][col]),tuple(gz_centers[row][col]),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

cv2.imshow("Solved",warped)
cv2.waitKey(0)

#process the src and dst points
pt_src = [[0,0],[warped.shape[1],0],[warped.shape[1],warped.shape[0]],[0,warped.shape[0]]]
pt_src = np.array(pt_src,dtype="float")
pt_dst = poly.reshape(4,2)
pt_dst = pt_dst.astype("float")

#align points in order
pt_src = order_points(pt_src)
pt_dst = order_points(pt_dst)

#calculate homography matrix
H,_ = cv2.findHomography(pt_src,pt_dst)

#reproject the puzzle to original image
im_out = cv2.warpPerspective(warped,H,dsize=(gray.shape[1],gray.shape[0]))
im_out = cv2.addWeighted(gray,0.9,im_out,0.2,0)

cv2.imshow("Projected",im_out)
cv2.waitKey(0)