# OCR-Dev
A small computer vision project in the making. The goal is to create a simple website that implements CNN Keras model to recognize handwritten characters (letters and digits) from an input file (a photo, image, or a scanned file). Beside CNN Keras, we are also using CRAFT (Character-Region Awareness For Text detection) in Pytorch. The point is to run our model on seperate character. 

## Progress
- We used the popular MNIST dataset with over 60k images of handwritten digits to practice using CNN in Keras. 
- Developed a more complexed CNN model on EMNIST (Extended-MNIST) dataset that returns 91.30% accuracy for EMNIST Bymerge and 88.20% for EMNIST Byclass.
- Current models are performing low accuracy on non-EMNIST testing dataset.
- Using CRAFT to extract character images from IAM dataset as a data augmentation method.
- Training a new model on the new dataset.
- Implementing prebuilt tools to detect lines in texts.
- Implementing prebuilt tools to convert italic texts to straight texts.


## Partners 
- Minh Quan Huynh: https://github.com/hmq1812
- Duc Minh Hoang: https://github.com/minhduchoang301/OCR-community-website
