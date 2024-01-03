import zipfile
import os

with zipfile.ZipFile("fra-eng.zip", "r") as zip_ref:
    zip_ref.extractall("data")

# Load the data
with open("data/fra.txt", "r", encoding="utf-8") as file:
    lines = file.read().split("\n")

input_texts = []
target_texts = []

for line in lines[: min(10000, len(lines) - 1)]:
    input_text, target_text, _ = line.split("\t")
    input_texts.append(input_text)
    target_texts.append(target_text)

# Tokenization
input_texts = [input_text.split() for input_text in input_texts]
target_texts = [target_text.split() for target_text in target_texts]
