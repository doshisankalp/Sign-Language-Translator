# Overview


#Abstract:
	A sign language translator in a real-time is an important climacteric in communication between the deaf-mute
community and normal public. Speech-Impaired people communicate among themselves using Sign Language but to normal
people, it is quite challenging to understand. Therefore, we hereby present the approach which is developed with the help of
visual-based method. We utilize MobileNets (open source model for efficient On-Device vision) to apply transfer learning to the
supervised machine learning model for image classification. The machine learning model is build using Tensorflow and image
processing is done with the help of a python library named OpenCV. The purpose of this project is to develop a web application
for Sign Language translation which will allow a normal person to translate Indian Sign Language (ISL) in a textual format which
includes displaying most probable word from the series of output alphabet and vice versa.

Introduction:-
	Communication among two individuals is a basic part of the social fabric of the society. However some individual unfortunately have to achieve this the hard way. Among them are the people of the speech impaired community. However they also need to interact with the people who do not have this impairment. But there is a problem in this case in which a normal person will feel very hard to understand the speech impaired person’s sign language which they use to communicate. Due to this there exists a possibility that these people may get isolated at workplaces or other public forums.
	The main aim of the project is to identify the alphabet in Indian Sign Language (ISL) from the input video using a web camera. The Gesture recognition and sign language recognition is a well-researched topic for ASL, and very less research work is published regarding Indian Sign Language (ISL). It is not convenient to always use a glove or Kinect to communicate and hence this idea came up with solving this problem using computer vision and CNN.
This project includes four tasks to be done in real time:
1. Taking input of real-time video consisting of Indian Sign Language (input).
2. Distinguish each frame from the video input.
3. Predicting and displaying most likely word from series of alphabets.
4. Acquiring text input from the user and to display its respective ISL gestured animation.

