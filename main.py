import cv2
import numpy as np
import copy
import tensorflow as tf
import time
import os
import sys
import autocomplete

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))

def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def suggest(alp):
	autocomplete.load()
	l=autocomplete.predict('the',alp)
	i=0
	while(i<5 and i<len(l)):
		print(l[i][0])
		i=i+1

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  #file_reader = tf.read_file(file_name, input_name)
  image_reader = file_name
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


opt=input("Welcome to Sign Language Translator\n1.Convert Sign to Text\n2. Convert Text to Sign\nEnter your option:")
if opt=="1":
	cam=cv2.VideoCapture(0)
	cam.set(200,200)

	model_file = "tf_files/retrained_graph.pb"
	label_file = "tf_files/retrained_labels.txt"
	input_height = 224
	input_width = 224
	input_mean = 128
	input_std = 128
	input_layer = "input"
	output_layer = "final_result"
	graph = load_graph(model_file)

	threshold = 19  #  BINARY threshold
	blurValue = 41  # GaussianBlur parameter
	bgSubThreshold = 70
	learningRate = 0

	cv2.namedWindow('trackbar')
	cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)
	count=0
	#print(type(count))

	lower = np.array([0, 0, 0], dtype = "uint8")
	upper = np.array([74, 255, 255], dtype = "uint8")

	# variables
	isBgCaptured = 0   # bool, whether the background captured
	triggerSwitch = False  # if true, keyborad simulator works

	cap_region_x_begin=0.5  # start point/total width
	cap_region_y_end=0.8  # start point/total width
	output=""

	while cam.isOpened():
		ret,frame=cam.read()
		frame = cv2.bilateralFilter(frame, 5, 50, 100) 
		frame = cv2.flip(frame, 1)
		cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
		         (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
		cv2.imshow("original1",frame)
		frame = frame[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
		frameee=frame
		#print(count)

		if isBgCaptured == 1:  # this part wont run until background captured
			count=(count+1)%110
			if(count==109):
				img = removeBG(frame)
				converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
				skinMask = cv2.inRange(converted, lower, upper)

					# apply a series of erosions and dilations to the mask
					# using an elliptical kernel
				kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
				skinMask = cv2.erode(skinMask, kernel, iterations = 2)
				skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

				# blur the mask to help remove noise, then apply the
				# mask to the frame
				skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)
				skin = cv2.bitwise_and(img, img, mask = skinMask)

				# show the skin in the image along with the mask
				#cv2.imshow("images", img)
				# convert the image into binary image
				gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
				blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
				#cv2.imshow('blur', blur)
				ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
				cv2.imshow('ori', thresh)
				
				if(cv2.countNonZero(thresh)<13000):
					print("Please keep the Hand in frame")
				else:
					# get the coutours
					thresh1 = copy.deepcopy(thresh)
					_,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
					length = len(contours)
					maxArea = -1
					if length > 0:
						for i in range(length):  # find the biggest contour (according to area)
							temp = contours[i]
							area = cv2.contourArea(temp)
							if area > maxArea:
								maxArea = area
								ci = i

						res = contours[ci]
						hull = cv2.convexHull(res)
						drawing = np.zeros(img.shape, np.uint8)
						cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
						#cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

						#app('System Events').keystroke(' ')  # simulate pressing blank space


					cv2.imshow('output', drawing)

					t = read_tensor_from_image_file(drawing,
				        			  input_height=input_height,
				        			  input_width=input_width,
				        			  input_mean=input_mean,
				        			  input_std=input_std)

					input_name = "import/" + input_layer
					output_name = "import/" + output_layer
					input_operation = graph.get_operation_by_name(input_name);
					output_operation = graph.get_operation_by_name(output_name);

					with tf.Session(graph=graph) as sess:
						start = time.time()
						results = sess.run(output_operation.outputs[0],
						      {input_operation.outputs[0]: t})
						end=time.time()

					results = np.squeeze(results)

					top_k = results.argsort()[-5:][::-1]

					labels = load_labels(label_file)
					#print(labels)
					print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
					template = "{} (score={:0.5f})"
					if(labels[top_k[0]]==''):
						suggest(output1)
						break
					output=output+labels[top_k[0]]
					output1=output[1:]
					if(len(output1)>0):
						print(output1)
		#			print(template)
					#for i in top_k:
					#	print(template.format(labels[i], results[i]))





		k = cv2.waitKey(2)
		if k == 27:  # press ESC to exit
			break
		elif k == ord('b'):  # press 'b' to capture the background
			bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
			isBgCaptured = 1
			output=""
			output1=""
			print( '!!!Background Captured!!!')
		elif k == ord('r'):  # press 'r' to reset the background
			bgModel = None
			triggerSwitch = False
			isBgCaptured = 0
			print ('!!!Reset BackGround!!!')
		elif k == ord('n'):
			triggerSwitch = True
			print ('!!!Trigger On!!!')
elif opt=="2":
	input_str=input("Enter the String: ")
	input_str=input_str.upper()
	print("Entered string: "+input_str)
	for i in range(0,len(input_str)):
		strr="sample/"+input_str[i]+".jpg"
		image = cv2.imread(strr)
		if not image is None:
			cv2.imshow("Display 1", image)
			cv2.waitKey(1500)
		else:
			print(input_str[i]+" not found")

else:
	print("Please Enter correct input.!\nBye")
	sys.exit()

