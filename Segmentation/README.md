# Segmentation
This includes line segmentation and word segmentation 

## Line segmentation 
Our line segmentation code largely implements the line height (blank line) calculation from [CrazyCrud Github](https://github.com/CrazyCrud/simple-text-line-extraction). What we added to this code is to use the line height calculated and an image of black and white stripes called "Thresholded Projection Profile" to find the starting and ending points of lines of words. By this way, we have been able to crop the segment the lines. However, the accuracy level of this technique is not abosolute since it depends on a bias number that we use during the cropping process. This bias will extend the area from the starting point to the ending point mentioned above of the lines of words. To calculate the bias, we are considering using a simple linear regression model. For now, it can be concluded that in order to fully crop the words, larger line heights will require larger biases.

## Word segmentation
Our word segmentation code implements this [source code](https://github.com/githubharald/WordSegmentation) from Harald Scheidl. However, the accuracy in detecting words is not 100%. What we can conclude at the moment is straight and clean handwritting texts will provide a better segmentation.
