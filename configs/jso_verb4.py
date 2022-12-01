
import json
import itertools


def main():
    words = []

    with open("words_pos.json", "r") as read_file:
        words = json.load(read_file)

    for word in words:
        for pos in word["POS"]:
            val = pos["pos"][0:2]
            if val == "VB":
                pos["verb_type"] = ""

    with open("words_pos.json", "w") as write_file:
        json.dump(words, write_file)

main()
