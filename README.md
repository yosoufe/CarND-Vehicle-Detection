# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Car_NotCar.png "Car vs NotCar Sample"
[image2]: ./output_images/features.png "Visualisation of Features"
[image3]: ./output_images/actual_features.png "Raw Features"
[image4]: ./output_images/Normalized_features.png "Normalized Features"
[image5]: ./output_images/Normalizer.png "Normalizer"
[image6]: ./output_images/sliding_wins_1.png "Sliding Windows"
[image7]: ./output_images/sliding_wins_2.png "Faster Sliding Windows"
[image8]: ./output_images/detection_samples.png "More Example of Car Detection"
[image9]: ./output_images/heat_map_sample.png "Heat Map Sample"
[image10]: ./output_images/labels_samples.png "Thresholded labels samples"
[image11]: ./output_images/final_detection_sample.png "Final Detection Samples"
[video1]: ./project_video_out.mp4

I have used [this jupyter notebook](https://github.com/yosoufe/CarND-Vehicle-Detection/blob/master/Vehicle%20Detection.ipynb) to complete this project.

## 1. Car/Not_Car Classifier

In order to start this project, first step, a classifier is needed to discriminate cars and not car images. To train the classifier, defining features is essential. The following procedure is used to train a classifier.

1. Importing the Training Data

   First of all all the training images addresses are stored in lists and imported to the workspace. The following is a sample of each group. Because the examples have the png format and we are importing them with matplotlib package, their pixels have values between 0 and 1 instead of 0 and 255.
   
   ![alt text][image1]

2. Create a Feature Extraction Pipline

   Next step is to extract features from the images. There different set of features used for this project. First of all is raw color values of the image in RGB in 32 x 32 pixels which is called here `bin_spatial`, then histogram of the colors which is called `color_hist` and finally the HOG (or Histogram of Oriented Gradients).
   
   HOG is calculated using `get_hog_features` functions and it can be applied to different color spaces. I found using all three channels of HLS color space is more robust and would result in less false positive detections. The following image is a sample of features of both groups.
   
   ![alt text][image2]
   
   Now we need to normalize the features to get faster convergance in the training algorith. I am using the `StandardScaler` from the sklearn package. The following two images demonstrate a sample of raw features and normalised features.
   
   ![alt text][image3]
   
   ![alt text][image4]
   
   The Following image demonstrates the normaliser features. As you can see this shows that our feature extraction is not optimised because variance of some of the features are zero. The scale of some of the features are zeros. That means they are useless features. It might be better to remove some of these useless features to have a faster algorithm. 
   
   ![alt text][image5]
   
   Finally we need to randomised our normalized dataset suing the `shuffle` from sklearn package.

3. Apply the Feature Extraction Pipline to Dataset
4. Spliting Dataset to Train, Test and Validation Dataset
5. Train the Classifier with Different Parameters and Choose the Best and Evaluate and save it

   Now I am using the Support Vector Machine in sklearn package as classifier. First I am using `GridSearchCV` to find the best parameters for my classifier. I did this on different `C` coffecient. I am using linear kernel. 'rbf' decreases the detection algorithm a lot therefore I decided to use linear kernel. The GridSearchCV is deviding the data into train, validation and test dataset itself and score them. Therefore I do not need to care about splitting data.
   
   Finally I obtained the best parameters and retrained a seperate svm with the same parameters. Becasue the svm which can be obtained by `GridSearchCV.best_estimator_` is very slow to use and a seperate svm classifer object is much faster to use.
   
   All of these are done in subsection called `Train the Classifier with Different Parameters, Choose the Best, Evaluate and Save it` in my notebook.
   
   At the end I am saving the classifier and the normalizer into pickle files for further uses.

## Searching for cars on the image:
I started by creatiung single scale sliding windows and then developed that one on different scales. At the beggining I used the following windows to search on the image:

![alt text][image6]

The above windows made the algorithm very slow. Almost 10 seconds per frame. I decided to decrease the number windows. Therefore I used the following windows Because I knew in this video all cars are on right side of the image.

![alt text][image7]

Therefore the overlapping and scales are chosen based on the calculation time and the probability of the presence of a car with that scale on specific postions on the image. Because small cars are close to horizon and big cars are covering big area of the image.

I chosed a small overlap, 35%, to have a faster algorithm (fewer sliding windows) and I am compensating that by using heatmap method and information from previous frames to find the cars as good as possible. Which those algorithm are much faster.

![alt text][image8]

The above picture demonstrates of the detection of cars. As you can see although the overlap of the sliding windows are small, they are doing a good job. the only problem is that detection of cars far away are bit difficult and they are only detected few times. By chossing the information of previous frames, this problem can also be solved.

### Heatmap:
Creating a heatmap means, initialize a map (array) with the same size of the image, Increase the value of each pixel by one if they are in a window and do this check for all detected windows. 

## Final Result:
You can find the resulting video [here](https://youtu.be/LlQv3c4PjVg)

The function `detect_cars` is analysing the video.
To avoid false positives and also to compensate the small overlap of sliding windows, I have some global variables to accumulate the heatmap from last 18 frames and I keep the pixels with at least value of 10 in the accumulative heatmap.


### Some More Visualisation:

#### Here are some raw detection

![alt text][image8]

#### Here are the corresponding heatmap

![alt text][image9]

#### Here are the lables:
The heatmap is thresholded to keep pixels with value larger than 1.

![alt text][image10]

### Here is the final detection

![alt text][image11]
