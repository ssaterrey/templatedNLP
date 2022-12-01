
import json
import itertools


names = []

with open("names.json", "r", encoding="utf-8") as read_file:
    names = json.load(read_file)

for name in names:
    name["synonyms"] = []

with open("names.json", "w") as write_file:
    json.dump(names, write_file)
