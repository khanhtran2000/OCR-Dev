import numpy as np
import pandas as pd
import craft
import math
import cv2
from collections import defaultdict
import scipy.cluster.hierarchy as hcluster
import sklearn.cluster as cluster


class Preprocess():
    """To segment words from line and or chars from word
    ====================================================
    """

    def segment_all(self, path):
        # Define functions
        def load_image(path: str):
            img = cv2.imread(path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
        def invert(img):
            return cv2.bitwise_not(img)

        def binarize_image(img):   
            ret, img_binarized = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            return img_binarized

        def equalize_image(img):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(img)
                
        def get_projection_profile(img):
            return np.sum(img, 1)

        def threshold_projection_profile(projection):
            projection_scaled = np.interp(projection, (projection.min(), projection.max()), (0,1))
                
            threshold = 0.3
            threshold_indices = projection_scaled > threshold
                
            projection_scaled.fill(0)
            projection_scaled[threshold_indices] = 1
                
            return projection_scaled

        def calculate_line_height(projection):
            change_indices = np.where(projection[:-1] != projection[1:])[0]
                
            heights = []
            for (index, change_index) in enumerate(change_indices):
                change_index_prev = 0 if index == 0 else change_indices[index - 1]
                    
                if projection[change_index] == 1:
                    height = change_index - change_index_prev
                    heights.append(height)
                
            return np.mean(heights)
                
        def get_image_of_projection_profile(projection, img):
            projection = np.interp(projection, (projection.min(), projection.max()), (0,1))
            
            maximum = np.max(projection)
            width = img.shape[1]
            result = np.zeros((projection.shape[0], 500))
            
            for row in range(img.shape[0]):
                cv2.line(result, (0, row), (int(projection[row]*width/maximum), row), (255,255,255), 1)
                    
            return result
                
        def remove_connected_components(img):
            number_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
            sizes = stats[1:, -1] 
            number_of_labels = number_of_labels - 1
            min_size = 100 
                
            img_cleaned = np.full(img.shape, 0)
            for i in range(0, number_of_labels):
                if sizes[i] >= min_size:
                    img_cleaned[labels == i + 1] = 255
                        
            return img_cleaned

        def run(img):  
            if img is None:
                return
                
            img_equalized = equalize_image(img.copy())
                
            img_binarized =  binarize_image(img_equalized.copy())
            img_binarized = invert(img_binarized)
                
            img_denoised_cc = remove_connected_components(img_binarized.copy())
                
            projection_profile = get_projection_profile(img_denoised_cc)
                
            projection_profile_thresholded = threshold_projection_profile(projection_profile)
            img_profile_thresholded = get_image_of_projection_profile(projection_profile_thresholded, img_denoised_cc)
                
            line_height = calculate_line_height(projection_profile_thresholded)

            return [line_height, img_profile_thresholded]
        
        # Import image
        img = load_image(path)
        
        # Run run() function on the image
        line_height = run(img)[0]
        img_thresholded = run(img)[1]

        # Create a dataframe
        img_df = pd.DataFrame(data=img_thresholded[0:,1:],
                    index=[i for i in range(img_thresholded.shape[0])],
                    columns=["col_"+str(i) for i in range(1,img_thresholded.shape[1])])
            
        col_1 = img_df["col_1"].values

        # Indexes of where a new line starts and ends
        white_index = []
        for index, pixel in enumerate(col_1):
            if pixel == 0.0:
                if index < (len(col_1)-1):
                    if (col_1[index-1] == 255.0):
                        white_index.append(index)
                    elif (col_1[index+1] == 255.0):
                        white_index.append(index+1) 
            
        # This is not a constant but a rate. Apply linear regression to find the formula. 
        # Basically, the larger the line height, the larger the bias
        bias = line_height * 0.667 - 1

        cropped_lines = []

        # Line segmenting based on start point, end point, and bias.
        for i in range(0,len(white_index),2):
            start = white_index[i]
            end = white_index[i+1]

            cropped_line = img[start-bias:end+bias, :]
            cropped_lines.append(cropped_line)

        # First array is blank since the index starts at 0 
        del cropped_lines[0]

        return cropped_lines

        # Crop word from line function
    def crop_word_from_line(self, line_list):
        words_all_lines = []

        for line in line_list:
            line_cp1 = line.copy()
            line_cp2 = line.copy()
            line = np.dstack((line,line_cp1,line_cp2))

            bboxes, polys, heatmap = craft.detect_text(line)
            words_one_line = []
            for i in range(len(bboxes)):
                x1 = bboxes[i][0][0]
                y1 = bboxes[i][0][1]
                x2 = bboxes[i][1][0]
                y2 = bboxes[i][1][1]
                x3 = bboxes[i][2][0]
                y3 = bboxes[i][2][1]
                x4 = bboxes[i][3][0]
                y4 = bboxes[i][3][1]

                top_left_x = int(min([x1, x2, x3, x4]))
                top_left_y = int(min([y1, y2, y3, y4]))
                bot_right_x = int(max([x1, x2, x3, x4]))
                bot_right_y = int(max([y1, y2, y3, y4]))

                word = line[top_left_y:bot_right_y, top_left_x:bot_right_x]
                words_one_line.append(word)
        
            words_all_lines.append(words_one_line)

        return words_all_lines

    def crop_char_from_word(self, words_list):
        # All characters
        chars_all = []
        for words_one_line in words_list:
            # Characters of a line
            chars_line = []
            for word in words_one_line:
                if len(word) > 0:
                    chars_dict = {}
                    # Characters of a word
                    chars_list = []

                    bboxes, polys, heatmap = craft.detect_text(word, link_threshold = 99999999, text_threshold = 0.4)
                    for i in range(len(bboxes)):
                        x1 = bboxes[i][0][0]
                        y1 = bboxes[i][0][1]
                        x2 = bboxes[i][1][0]
                        y2 = bboxes[i][1][1]
                        x3 = bboxes[i][2][0]
                        y3 = bboxes[i][2][1]
                        x4 = bboxes[i][3][0]
                        y4 = bboxes[i][3][1]

                        top_left_x = int(min([x1, x2, x3, x4]))
                        top_left_y = int(min([y1, y2, y3, y4]))
                        bot_right_x = int(max([x1, x2, x3, x4]))
                        bot_right_y = int(max([y1, y2, y3, y4]))

                        char = word[top_left_y:bot_right_y, top_left_x:bot_right_x]

                        # Append a new key:value pair to the dictionary with top_left_x value as the key
                        chars_dict.update({top_left_x:char})

                    # Sort the keys, then append the corresponding values to letters_list
                    for key in sorted(chars_dict.keys()):
                        chars_list.append(chars_dict[key])
                
                chars_line.append(chars_list)
            chars_all.append(chars_line)
        
        return chars_all



