import json

class Acronym:
    def __init__(self):
        file_path = 'DatasetVniAcronym/Acronym.json'
        with open(file_path, 'r', encoding='utf-8') as json_file:
            self.my_dict = json.load(json_file)
    def Convert(self,value):
        return [key for key, val in self.my_dict.items() if value in val][0] if any(value in val for val in self.my_dict.values()) else value
    def Solve_Acr(self,string):
        words = string.split(' ')
        return ' '.join([self.Convert(word) for word in words])
