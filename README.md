# Hand-Gesture-Recognition

The Training.ipynb file is the file in which we can create the dataset and train the model. I have used Google's Mediapipe to implement this project, as in this scenario its a much better way to predict my Hand Gestures rather than using CNN or Convolutional Neural Networks. Mediapipe is library we can utilize to implement this, I have used the hands module  in this library. It has 21  points or landmarks of a hand when it predicts one, there are landmarks for thumb_tip, index_tip and so on to signify tip of the thumb and tip of the index finger respectively. By using the landmarks, that is the coordinates of the different landmarks, we can get a more accurate model rather than using CNN in this case.

CNNs in this case wouldn't work as well as they are sensitive to the background and also the lighting conditions. These minute changes can affect the prediction of the model. By using these andmarks, we can have a simple neural network and train it on just numbers.

In the training.ipynb file I first create the dataset, I have written the code such that my webcam feed opens and if i press a number key on my keyboard say 1  then it will start capturing the landmarks/coordinates of my hand in that frame, so if I hold the key then it will  capture all of the coordinates frame by frame. So I have just created a dataset like this by clicking a label/class and moving my hand in the frame and rotating and such to create a diverse dataset (I have basically augmented the gestures, I did also try to rotate the landmarks by code, but resorted not to at the end due to the nature of my gestures and their similarity).

After getting the seperate csv files for each gesture i concatenated them and created a one single dataframe. Then i have just use train test split to create the training and the validation datasets and then just feed t to the model to get fit.

After fitting the model and getting a good accuracy. We czn run the testing.py script to predict the hand gestures on live Camera feed. The model predicts the label/class for live feed from the webcam and its preety accurate.
