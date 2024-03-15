from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import random
import torch

import src.utils as utils

class LM:
    def __init__(self, entity_type, model_name = 'meta-llama/Llama-2-7b-hf', device="cuda:0") -> None:
        self.entity_names = {"LOC": "Location", 
                             "ORG": "Organization",
                             "MISC": "Miscellaneous",
                             "PER": "Person"}
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype= torch.float16).to(device)
        self.max_length = 4096
        self.device = device
        self.entity_type = entity_type.upper()

        
        
    def __inference(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        output = self.model.generate(input_ids, max_length=self.max_length)
        input_length = input_ids.shape[1]
        output = output[:, input_length:]

        output = self.tokenizer.decode(output[0])

        return output


    def self_verify(self, original_sentence, 
                    model_pred_sent, 
                    yes_examples, 
                    no_examples
                ):
        
        prompt = utils.SELF_VERIFICATION_INITIAL_TEMPLATE.format(self.entity_names[self.entity_type])
        examples = [(i, "Yes") for i in yes_examples] + \
                   [(i, "No") for i in no_examples]
        
        random.shuffle(examples)
        for example, response in examples:
            # prompt += utils.SELF_VERIFICATION_EXAMPLE_TEMPLATE\
            #             .format(example, )
            pass
    
    def do_ner(self, 
               sentence, 
               pos_examples, 
               neg_examples
               ):
        prompt = utils.NER_INITIAL_TEMPLATE.format(self.entity_names[self.entity_type])
        examples = []

        #relevant_examples
        for example in pos_examples:
            input_sent = example["sentence"]
            output_sent = example["tagged"]
            example = utils.NER_EXAMPLE_TEMPLATE.format(input_sent, output_sent)
            examples.append(example)

        #negative_examples
        for example in neg_examples:
            input_sent = example["sentence"]
            output_sent = example["sentence"]
            example = utils.NER_EXAMPLE_TEMPLATE.format(input_sent, output_sent)
            examples.append(example)

        random.shuffle(examples)

        # add the input sentence as final input
        examples.append(utils.NER_EXAMPLE_TEMPLATE.format(sentence, ""))

        examples = "".join(examples)

        prompt = prompt + examples

        output = self.__inference(prompt)

        return output
        



