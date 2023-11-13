import json

class Acronym:
    """
    A class for converting acronyms in a given text based on a predefined dictionary.
    """

    def __init__(self):
        """
        Initializes the Acronym class by loading the acronym dictionary from a JSON file.
        """
        file_path = 'dataset_vni_acronym/acronym.json'
        with open(file_path, 'r', encoding='utf-8') as json_file:
            self.my_dict = json.load(json_file)

    def Convert(self, value):
        """
        Converts the provided value to its corresponding key in the acronym dictionary.

        Args:
            value (str): The input value to be converted.

        Returns:
            str: The converted value (acronym) or the original value if not found in the dictionary.
        """
        return [key for key, val in self.my_dict.items() if value in val][0] if any(value in val for val in self.my_dict.values()) else value

    def Solve_Acr(self, string):
        """
        Solves acronyms in the given string by converting each word using the acronym dictionary.

        Args:
            string (str): The input string containing words to be processed.

        Returns:
            str: The modified string with acronyms replaced.
        """
        words = string.split(' ')
        return ' '.join([self.Convert(word) for word in words])
