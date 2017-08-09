#2017/07/31


##---getting the data to X_train and y_train

import csv
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cv2 import cvtColor

from os import path
file_path = path.relpath('data_new/')

lines = []
with open(file_path + '/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#----create a list of images and measurements with original and argumented (flipped) data
#----Use of the "left" and "right" images and correct them by offset and factor


def get_img_and_meas(source_path, corr, fac):
    meas_temp=[]
    img_temp=[]

    filename = source_path.split('/')[-1]
    current_path=file_path+'/IMG/' + filename
    img = cv2.imread(current_path)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_temp.append(image)
    image_flipped = np.fliplr(image)
    img_temp.append(image_flipped)
    measurement=(float(line[3])+corr)*fac
    meas_temp.append(measurement)
    measurement_flipped = -measurement
    meas_temp.append(measurement_flipped)

    return (img_temp, meas_temp)


images = []
measurements = []

print (len(lines))

for line in lines:
    source_path_center =line[0]
    (img_temp, meas_temp) = get_img_and_meas(source_path_center, corr=0, fac=1.0)
    images.extend(img_temp)
    measurements.extend(meas_temp)

    source_path_right = line[2]
    (img_temp, meas_temp) = get_img_and_meas(source_path_right, corr=-0.2, fac=1.0)
    images.extend(img_temp)
    measurements.extend(meas_temp)

    source_path_left = line[1]
    (img_temp, meas_temp) = get_img_and_meas(source_path_left, corr=0.2, fac=1.0)
    images.extend(img_temp)
    measurements.extend(meas_temp)

print (len(images))
print (len(measurements))

X_train = np.array(images)
y_train = np.array(measurements)

print (X_train.shape)
print (y_train.shape)

#----plot histogram to see the distribution of the steering angle values

# num_bins = 30
# avg_samples_per_bin = len(measurements)/num_bins
# hist, bins = np.histogram(measurements, num_bins)
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist, align='center', width=width)
# plt.plot((np.min(measurements), np.max(measurements)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
# plt.show()
# print (hist)
# print (bins)

#----create a target, based on the average value to delete/remove the dominant number of 0° steering angle

# keep_probs = []
# target = avg_samples_per_bin * 1
# for i in range(num_bins):
#     if hist[i] < target:
#         keep_probs.append(1.)
#     else:
#         keep_probs.append(1./(hist[i]/target))
# remove_list = []
# for i in range(len(measurements)):
#     for j in range(num_bins):
#         if measurements[i] > bins[j] and measurements[i] <= bins[j+1]:
#             # delete from X and y with probability 1 - keep_probs[j]
#             if np.random.rand() > keep_probs[j]:
#                 remove_list.append(i)
# images_red = np.delete(images, remove_list, axis=0)
# measurements_red = np.delete(measurements, remove_list)
#
#
# hist, bins = np.histogram(measurements_red, num_bins)
# plt.bar(center, hist, align='center', width=width)
# plt.plot((np.min(measurements), np.max(measurements)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
# plt.show()
# print (hist)
# print (bins)
#
# X_train = np.array(images_red)
# y_train = np.array(measurements_red)


##---Create Nvidia-NN with following adjustments: adding MaxPoling and Dropout to reduce the overfitting problem

from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Cropping2D
from keras.optimizers import Adam

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation='elu'))
#model.add(Dropout(0.5))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam(lr=1e-4))
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')
