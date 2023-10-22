import string

def remove_punctuation(input_string):
    translator = str.maketrans('', '', string.punctuation)
    cleaned_string = input_string.translate(translator)
    return cleaned_string

# # File paths
# input_file = fr"DataSet\all\sents.txt"
# output_file = fr"DataSet\all\cleaned_dataset.txt"
#
# # Read data from the input file
# with open(input_file, 'r', encoding='utf-8') as f:
#     sentences = f.readlines()
#
# # Process each sentence and remove punctuation
# cleaned_sentences = [remove_punctuation(sentence.strip()) for sentence in sentences]
#
# # Save the cleaned sentences to the output file
# with open(output_file, 'w', encoding='utf-8') as f:
#     for sentence in cleaned_sentences:
#         f.write(sentence + '\n')
#
# print("Punctuation removed and cleaned sentences saved to 'cleaned_dataset.txt'.")
