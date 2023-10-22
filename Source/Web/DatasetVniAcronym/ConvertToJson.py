import json
with open('Target.txt',encoding='utf8') as f:
    keys = f.readlines()
keys = [key.strip() for key in keys]
dic_acr ={}
file = open('Acronym.txt',encoding='utf8')
for key in keys:
    values = file.readline()
    dic_acr[key] = values[:len(values)-1].split(',')
file.close()
print(dic_acr)
# Example dictionary
my_dict = dic_acr
# Specify the file path where you want to save the dictionary as JSON
file_path = 'Acronym.json'
# Open the file in write mode and save the dictionary as JSON
with open(file_path, 'w', encoding='utf-8') as json_file:
    json.dump(my_dict, json_file, ensure_ascii=False)
print("Dictionary saved to JSON file successfully.")
