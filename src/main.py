import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data.dataset import SignalDataset
model_name = 'VMware/open-llama-7B-open-instruct'
SIGNAL_DIR = "Signal_1M"

ds = SignalDataset(SIGNAL_DIR)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype= torch.float16).to("cuda")

GUIDELINES_PROMPT = (
    "Entity Definition:\n"
    "1. PERSON: Short name or full name of a person from any geographic regions.\n"
    "2. ORG: Companies, political groups, musical bands, sport clubs, government bodies, and public organizations. Nationalities and religions are not included in this entity type.\n"
    "3. LOC: Name of any geographic location, like cities, countries, continents, districts etc.\n"
    "\n"
    "Output Format:\n"
    "{{'PERSON': [list of entities present], 'ORG': [list of entities present], 'LOC': [list of entities present]}}\n"
    "If no entities are presented in any categories keep it None\n"
    "\n"
    "Examples:\n"
    "\n"
    "1. Sentence: Mr. Jacob lives in Madrid and works at Microsoft\n"
    "Output: {{'PERSON': ['Mr. Jacob'], 'ORG': ['Microsoft'], 'LOC': ['Madrid']}}\n"
    "\n"
    "2. Sentence: Mr. Rajeev Mishra and Sunita Roy are friends and they meet each other in front of Aamazon office.\n"
    "Output: {{'PERSON': ['Mr. Rajeev Mishra', 'Sunita Roy'], 'ORG': ['Amazon'], 'LOC': ['None']}}\n"
    "\n"
    "3. Sentence: {}\n"
    "Output: "
)

my_sentence = ds[0]['content']

inputt = GUIDELINES_PROMPT.format(my_sentence)


input_ids = tokenizer(inputt, return_tensors="pt").input_ids.to("cuda")

output1 = model.generate(input_ids, max_length=2048)
input_length = input_ids.shape[1]
output1 = output1[:, input_length:]
output= tokenizer.decode(output1[0])

print(output)


