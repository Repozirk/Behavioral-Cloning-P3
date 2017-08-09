#**Behavioral Cloning** 

##Writeup Template

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./center.png "Center Camera"
[image2]: ./center_flipped.png "Center Camera flipped"
[image3]: ./right.png "Right Camera"
[image4]: ./left.png "Left Camera"
[image5]: ./distriubution.png "Distribution of Training_Data"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

Starting with the propsal of the Nvidia CNN, deveoloped of the Nvidia Autonomous Car Team, I decided to change the architecture for better performance of the CNN on my dataset.

The dataset was splitted in test data and validation data.

Following changes have been done, to get the best results out of my data:
- adding a Lambda-Layer for data normalization, as suggested
- adding a Cropping-Layer to get the interesting area of the images
- adding MaxPooling-Layers after the Conv-Layers for better performance of the CNN
- adding a DropOut-Layer just before the first fully connecete layer to avoid overfitting
- use ELU instead of RELU as an acitvation function 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was tuned manually to 0.0001.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving clockwise and counter clockwise, recovering from the left and right sides of the road and driving the difficult curves of the track again.

The raw data consists of the images from the center, right and left images. I decided to work with all three cameras to get a as many raw data as possible and to avoid the concentration of data with steering angle=0°. For this it was necessary to correct the steeering angle for the right and left images.
![alt text][image1]
![alt text][image3]
![alt text][image4]

![alt text][image5]: ./distriubution.png "Distribution of Training_Data"


To improve the CNN performance on test and validation data, argumented data was created by flipping the images and taking the opposite sign of the steering angles.

![alt text][image1]
![alt text][image2]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 with an adam optimizer.
