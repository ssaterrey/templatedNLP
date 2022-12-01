
import json
import itertools


adjectives = []

with open("adjectives_use.json", "r") as read_file:
    adjectives = json.load(read_file)

for adj in adjectives:
    adj["result_of"] = ""

with open("adjectives_use.json", "w") as write_file:
    json.dump(adjectives, write_file)
