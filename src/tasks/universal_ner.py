import argparse
import os
import random
from src.utils import NER_TYPES, sentence_tokenize, load_conll_corpuses
from src.data.signal_dataset import SignalDataset
from src.language_model import NERModel


def main(args):
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    signal_ds = SignalDataset(args.signal_dir)
    WORKING_ENTITY = args.entity_type.upper()
    language_model = NERModel(WORKING_ENTITY)
    

    for signal_idx in range(len(signal_ds)):
        instance = signal_ds[signal_idx]
        sentences = sentence_tokenize(instance["content"])
        ners = language_model.do_ner(sentences)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--signal_dir", help="Directory of Signal 1M Dataset", required=True)
    parser.add_argument("--entity_type", choices=["per", "org", "loc", "misc"], required=True)
    parser.add_argument("--seed", default=6556)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    main(args)