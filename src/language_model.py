import fire
from vllm import LLM, SamplingParams

from transformers import LlamaTokenizer
import re
import random
import torch

from src.utils import preprocess_instance, get_response

class NERModel:
    def __init__(self, entity_type, model_path = 'Universal-NER/UniNER-7B-type', device="cuda:0") -> None:
        self.entity_names = {"LOC": "Location Name Entities", 
                             "ORG": "Organization Name Entities",
                             "MISC": "Miscellaneous Name Entities",
                             "PER": "Person Name Entities"}
        
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LLM(model=model_path, tensor_parallel_size=1)
        self.max_input_length = 1024
        self.max_new_tokens = 256
        self.device = device
        self.entity_type = entity_type.upper()

        
        
    def __inference(self, examples):
        prompts = [preprocess_instance(example['conversations']) for example in examples]
        sampling_params = SamplingParams(temperature=0, max_tokens=self.max_new_tokens, stop=['</s>'])
        responses = self.model.generate(prompts, sampling_params, use_tqdm=False)
        responses_corret_order = []
        response_set = {response.prompt: response for response in responses}
        for prompt in prompts:
            assert prompt in response_set
            responses_corret_order.append(response_set[prompt])
        responses = responses_corret_order
        outputs = get_response([output.outputs[0].text for output in responses])
        return outputs

    
    def do_ner(self, sentences
               ):
        examples = list()
        for sentence in sentences:
            examples.append({"conversations": 
                        [
                            {"from": "human", 
                            "value": f"Text: {sentence}"
                            }, 

                            {"from": "gpt", 
                            "value": "I've read this text."
                            }, 
                            
                            {"from": "human", 
                            "value": f"What describes {self.entity_names[self.entity_type]} in the text?"
                            }, 
                            
                            {"from": "gpt", 
                            "value": "[]"
                            }
                        ]
                        })
        output = self.__inference(examples=examples)
        return output