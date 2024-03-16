import argparse
import os
import random
import json
from tqdm import tqdm

from src.utils import NER_TYPES, sentence_tokenize, load_conll_corpuses
from src.data.signal_dataset import SignalDataset
from src.language_model import NERModel


def main(args):
    random.seed(args.seed)
    WORKING_ENTITY = args.entity_type.upper()
    os.makedirs(os.path.join(args.output_dir, WORKING_ENTITY), exist_ok=True)
    signal_ds = SignalDataset(args.signal_dir)
    language_model = NERModel(WORKING_ENTITY)
    

    for signal_idx in tqdm(range(0, len(signal_ds), args.batch_size)):
        sentences = []
        indices = {}
        for i in range(args.batch_size):
            if i + signal_idx >= len(signal_ds):
                break
            instance = signal_ds[signal_idx + i]
            sent_tokenized = sentence_tokenize(instance["content"])
            indices[instance["id"]] = (len(sentences), len(sentences) + len(sent_tokenized))
            sentences.extend(sent_tokenized)
        ners = language_model.do_ner(sentences)

        for key, value in indices.items():
            file_path = os.path.join(args.output_dir, WORKING_ENTITY, key)
            result = [] 
            start, end = value
            for i in range(start, end):
                result.append(ners[i])
            
            with open(file_path, "w") as f:
                json.dump(result, f)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--signal_dir", help="Directory of Signal 1M Dataset", required=True)
    parser.add_argument("--entity_type", choices=["per", "org", "loc", "misc"], required=True)
    parser.add_argument("--seed", default=6556)
    parser.add_argument("--batch_size", default=128)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    main(args)