import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data.dataset import SignalDataset
import nltk
import re

model_name = 'VMware/open-llama-7B-open-instruct'
SIGNAL_DIR = "../Signal_1M"

def self_verify(sentence, model_pred_sent, model, tokenizer):
    SELF_VERIFICATION_PROMPT = (
        "I am an excellent linguist. The task is to verify whether the word is a Location entity extracted from the given sentence.\n"
        "The given sentence: Only France and Britain backed Fischler's proposal.\n"
        "Is the word \"Britain\" in the given sentence a Location entity? Please answer with Yes or No.\n"
        "Yes.\n"
        "\n\n"
        "The given sentence: It brought in 4275 tonnes of British mutton. Some 10 percent of overall imports.\n"
        "Is the word \"British\" in the given sentence a Location entity? Please answer with Yes or No.\n"
        "Yes.\n"
        "\n\n"
        "The given sentence: {}\n"
        "Is the word \"{}\" in the given sentence a Location entity? Please answer with Yes or No.\n"
    )
    found_entities = re.findall(pattern=r"@@(.*?)##", string=model_pred_sent)
    for entity in found_entities:
        inputt = SELF_VERIFICATION_PROMPT.format(sentence, entity)
        input_ids = tokenizer(inputt, return_tensors="pt").input_ids.to("cuda")
        output1 = model.generate(input_ids, max_length=1024)
        input_length = input_ids.shape[1]
        output1 = output1[:, input_length:]
        decoded_out= tokenizer.decode(output1[0])
        if "yes" in decoded_out.lower():
            continue
        elif "no" in decoded_out.lower():
            model_pred_sent = model_pred_sent.replace(f"@@{entity}##", entity)
    return model_pred_sent

ds = SignalDataset(SIGNAL_DIR)
print(ds[1])
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype= torch.float16).to("cuda")

# GUIDELINES_PROMPT = (
#     "Entity Definition:\n"
#     "1. PERSON: Short name or full name of a person from any geographic regions.\n"
#     "2. ORG: Companies, political groups, musical bands, sport clubs, government bodies, and public organizations. Nationalities and religions are not included in this entity type.\n"
#     "3. LOC: Name of any geographic location, like cities, countries, continents, districts etc.\n"
#     "\n"
#     "Output Format:\n"
#     "{{'PERSON': [list of entities present], 'ORG': [list of entities present], 'LOC': [list of entities present]}}\n"
#     "If no entities are presented in any categories keep it None\n"
#     "\n"
#     "Examples:\n"
#     "\n"
#     "1. Sentence: Mr. Jacob lives in Madrid and works at Microsoft\n"
#     "Output: {{'PERSON': ['Mr. Jacob'], 'ORG': ['Microsoft'], 'LOC': ['Madrid']}}\n"
#     "\n"
#     "2. Sentence: Mr. Rajeev Mishra and Sunita Roy are friends and they meet each other in front of Aamazon office.\n"
#     "Output: {{'PERSON': ['Mr. Rajeev Mishra', 'Sunita Roy'], 'ORG': ['Amazon'], 'LOC': ['None']}}\n"
#     "\n"
#     "3. Sentence: {}\n"
#     "Output: "
# )
GUIDELINES_PROMPT = (
    "I am an excellent linguist. The task is to label location entities in the given sentence. Below are some examples\n"
    "Input: Only France and Britain backed Fischler's proposal.\n"
    "Output: Only @@France## and @@Britain## backed Fischler's proposal.\n"
    "\n"
    "Input: Germany imported 47600 sheep from Britain last year, nearly half of total imports.\n"
    "Output: @@Germany## imported 47600 sheep from @@Britain## last year, nearly half of total imports.\n"
    "\n"
    "Input: It brought in 4275 tonnes of British mutton. Some 10 percent of overall imports.\n"
    "Output: It brought in 4275 tonnes of British mutton. Some 10 percent of overall imports.\n"
    "\n"
    "Input: {}\n"
    "Output: "
)
outputs = []
for sentence in nltk.sent_tokenize(ds[1]["content"]):
    my_sentence = sentence
    inputt = GUIDELINES_PROMPT.format(my_sentence)
    input_ids = tokenizer(inputt, return_tensors="pt").input_ids.to("cuda")
    output1 = model.generate(input_ids, max_length=768)
    input_length = input_ids.shape[1]
    output1 = output1[:, input_length:]
    output= tokenizer.decode(output1[0])
    # print(my_sentence)
    output = self_verify(sentence, output, model, tokenizer)
    outputs.append(output)

with open("out.txt", "w") as f:
    outputs = [o + "\n" for o in outputs]
    f.writelines(outputs)



