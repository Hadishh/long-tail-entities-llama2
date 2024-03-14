import argparse
import os
import random
from src.utils import NER_TYPES, sentence_tokenize, load_conll_corpuses
from src.data.signal_dataset import SignalDataset
from src.language_model import LM

def unify_ner_examples(args, working_entity_type, sentence_idx, semantic_relations, corpuses):
    pos_examples = semantic_relations[working_entity_type][sentence_idx]
    pos_examples = random.choices(pos_examples, k=args.pos_examples_count)
    neg_examples = []
    examples_per_type_count = args.neg_examples_count // (len(NER_TYPES) - 1)
    for type_ in NER_TYPES:
        if type_ == working_entity_type:
            continue
        potential_neg_examples = semantic_relations[type_][sentence_idx]
        approved_neg_examples = []
        for example_id in potential_neg_examples:
            if type(corpuses[working_entity_type].get_by_id(example_id)) == type(str()):
                approved_neg_examples.append(example_id)
        
        approved_neg_examples = random.choices(approved_neg_examples, k = examples_per_type_count)

        neg_examples.extend(approved_neg_examples)
    
    return pos_examples, neg_examples

def main(args):
    corpuses = load_conll_corpuses()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    signal_ds = SignalDataset(args.signal_dir, semantics_dir=args.semantic_dir)
    WORKING_ENTITY = args.entity_type.upper()
    language_model = LM(WORKING_ENTITY)
    

    for signal_idx in range(len(signal_ds)):
        instance = signal_ds[signal_idx]
        sentences = sentence_tokenize(instance["content"])
        for idx, sentence in enumerate(sentences):
            semantic_relations = instance["semantic_relations"]
            pos_examples, neg_examples = unify_ner_examples(args, WORKING_ENTITY, idx, semantic_relations, corpuses)
            tagged_sentence = language_model.do_ner(sentence=sentence, pos_examples=pos_examples, neg_examples=neg_examples)

            if args.self_verify:
                tagged_sentence = language_model.self_verify(sentence, tagged_sentence, pos_examples, neg_examples)
            

                



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--signal_dir", help="Directory of Signal 1M Dataset", required=True)
    parser.add_argument("--semantic_dir", help="Directory of Signal 1M Dataset Semantic related intstances, gathered from semantic search.", required=True)
    parser.add_argument("--entity_type", choices=["per", "org", "loc", "misc"], required=True)
    parser.add_argument("--self_verify", type=bool, default=True)
    parser.add_argument("--pos_examples_count", default=24)
    parser.add_argument("--neg_examples_count", default=24)
    parser.add_argument("--seed", default=6556)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    main(args)