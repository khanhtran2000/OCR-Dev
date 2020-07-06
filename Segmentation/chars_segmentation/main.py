from Prepocessing_word_char import Preprocess

def main(input_dir: str, output_dir: str):
    preprocess = Preprocess()

    # Lines segmenting
    lines = preprocess.segment_all(input_dir)
    # Words segmenting
    words = preprocess.crop_word_from_line(lines)
    # Characters segmenting 
    chars = preprocess.crop_char_from_word(words)

    # Export character images
    line_num = 0
    for line in chars:

        word_num = 0
        for word in line:
            
            j = 0 
            for char in word:
                cv2.imwrite(output_dir + "/line{0}_word{1}_char{2}.png".format(line_num, word_num, str(j)), char)
                j += 1
                
            word_num += 1
        line_num += 1


if __name__ == '__main__':
    input_dir = input("Image path: ")
    output_dir = input("Output path: ")
	main(input_dir, output_dir)


