
import json
import itertools


nouns = []

with open("nouns_use.json", "r") as read_file:
    nouns = json.load(read_file)

for noun in nouns:
    noun["category"] = ""

with open("nouns_use.json", "w") as write_file:
    json.dump(nouns, write_file)
