# Using CRAFT on IAM dataset

## Problem and Solution
In order to create more variability for our model, we sought for more handwritten characters datasets but found very few resources. We did find some, but they were in other languages. In order to solve this problem, we resorted to using CRAFT to extract characters from images of handwritten words. We chose the IAM dataset to perform this task on. 

## Result
We were unable to extract all of the characters in the words, perhaps because of the imperfection in our CRAFT model. The majority of the characters were extracted from clearly written words. Words that had letters stick together provided too little space between letters to draw bounding boxes on, and so we drop those words, too. 
