from transformers import RobertaTokenizer
from typing import List
import os

from src.spel.configuration import device, get_local_model_path
from src.spel.data_loader import BERT_MODEL_NAME
from src.spel.model import SpELAnnotator
from src.spel.utils import get_subword_to_word_mapping
from src.spel.span_annotation import WordAnnotation, PhraseAnnotation
from src.utils import preprocess_instance, get_response

class SpELModel:
    def __init__(self) -> None:
        self.__finetuned_after_steps = 4
        self.tokenizer = RobertaTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.spel = SpELAnnotator()

        self.spel.init_model_from_scratch(device=device)

        if self.__finetuned_after_steps == 3:
            self.spel.shrink_classification_head_to_aida(device=device)
        
        self.spel.load_checkpoint( 
                                  get_local_model_path(), 
                                  device=device, 
                                  load_from_torch_hub=False, 
                                  finetuned_after_step=self.__finetuned_after_steps
                                )
    
    def __run_spel(self, sentence) -> List[PhraseAnnotation]:
        inputs = self.tokenizer(sentence, return_tensors="pt")
        token_offsets = list(zip(inputs.encodings[0].tokens,inputs.encodings[0].offsets))

        subword_annotations = self.spel.annotate_subword_ids(inputs.input_ids, k_for_top_k_to_keep=10, token_offsets=token_offsets)

        # CREATE WORD-LEVEL ANNOTATIONS
        tokens_offsets = token_offsets[1:-1]
        subword_annotations = subword_annotations[1:]
        word_annotations = [WordAnnotation(subword_annotations[m[0]:m[1]], tokens_offsets[m[0]:m[1]])
                            for m in get_subword_to_word_mapping(inputs.tokens(), sentence)]

        # CREATE PHRASE-LEVEL ANNOTATIONS
        phrase_annotation = []
        for w in word_annotations:
            if not w.annotations:
                continue
            if phrase_annotation and phrase_annotation[-1].resolved_annotation == w.resolved_annotation:
                phrase_annotation[-1].add(w)
            else:
                phrase_annotation.append(PhraseAnnotation(w))
        return phrase_annotation

    def do_el(self, sentences):
        annotations = []
        for sentence in sentences:
            phrase_annotation = self.__run_spel(sentence)
            print([pa.resolved_annotation for pa in phrase_annotation])


if __name__ == "__main__":
    spel = SpELModel()
    sentence = "Ronaldo in Brazil missed the corner."
    spel.do_el([sentence])

