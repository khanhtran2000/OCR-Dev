# EMNIST Recognition Using CNN Keras

## Files
- Model and weights saved in .pb and .h5 files. This model and its weights will produce 91.30% accuracy for an EMNIST test file. The model was trained on EMNIST bymerge with over 620k training images. 
- `Image Processing and Loading Model.ipynb` contains a function that convert your own images to MNIST (and EMNIST) format. The character or digit will be placed in the center of the image. The background will be black, and the character or digit will be white. Also reshape it to (28,28). 

## EMNIST Notebook
- Here are some guides for you to better understand `Fork of EMNIST_Test_Kaggle.ipynb`:
  * I used Kaggle to run the notebook so I basically did not download the datasets to my local computer. That is the directory has the format '../input/').
  * `reduce_mem_usage()` is a function to reduce the amount of RAM used for loading datasets into the notebook. Since Kaggle only allows 13GB of RAM, and Google Colab gives 16GB in its free version, it is not enough for loading the datasets and the training the model afterwards. 
  * It takes quite some time to load the datasets this way. But the amount of RAM saved is incredible. 
  * Kaggle supports Nvida Tesla P100 GPU which is much faster than Google Colab's Nvidia Tesla K80. However, that is in exchange of 3GB of less RAM, and sometimes the notebook can crash before you can test the model.
  * You can leave out `datagen` since data augmentation seems to hurt the performance in this case.
  * The model follows this [architecture](https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist). 
  * There are some uncommented cells. They do work but sometimes running them before the testing will use up all the RAM. 
- Visit my Kaggle profile for the full notebook: https://www.kaggle.com/christranvn/emnist-test-kaggle

## Dataset
Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373
