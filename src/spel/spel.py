from transformers import AutoTokenizer
from typing import List
import os

from src.spel.configuration import device, get_local_model_path
from src.spel.data_loader import BERT_MODEL_NAME, dl_sa
from src.spel.model import SpELAnnotator
from src.spel.utils import get_subword_to_word_mapping
from src.spel.span_annotation import WordAnnotation, PhraseAnnotation

class SpELModel:
    def __init__(self) -> None:
        self.__finetuned_after_steps = 4
        self.max_input_tokens = 512
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
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
    
        dl_sa.set_vocab_and_itos_to_all()
    
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
            data_ = []
            for pa in phrase_annotation:
                if pa.resolved_annotation <= 1:
                    #either pad or none
                    continue
                
                mention = dl_sa.mentions_itos[pa.resolved_annotation]
                span = (pa.begin_character, pa.end_character)
                phrase = pa.word_string

                data_.append({"mention": mention, "span": span, "phrase": phrase})

            annotations.append(data_)
        
        return annotations



if __name__ == "__main__":
    spel = SpELModel()
    sentences = ["Ronaldo scored 15 goals for Brazil in World Cup 1994."]
    spel.do_el(sentences)

