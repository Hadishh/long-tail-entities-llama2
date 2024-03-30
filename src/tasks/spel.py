import argparse
import os
import random
import json
from tqdm import tqdm

from src.utils import NER_TYPES, sentence_tokenize
from src.data.signal_dataset import SignalDataset
from src.spel.spel import SpELModel


def main(args):
    random.seed(args.seed)
    os.makedirs(os.path.join(args.output_dir, "ERRORS"), exist_ok=True)
    signal_ds = SignalDataset(args.signal_dir)
    entity_linker = SpELModel()
    
    failures = 0
    signal_chosen = random.sample(range(len(signal_ds)), k=256000)
    for batch_idx in (pbar := tqdm(range(0, len(signal_chosen), args.batch_size))):
        sentences = []
        indices = {}
        for i in range(args.batch_size):
            pbar.set_description(f"Failures: {failures}")
            if i + batch_idx >= len(signal_chosen):
                break
            signal_idx = signal_chosen[batch_idx + i]
            instance = signal_ds[signal_idx]
            sent_tokenized = sentence_tokenize(instance["content"])
           
            able_to_tokenize = True
            for sent in sent_tokenized:
                if (len(entity_linker.tokenizer(sent )['input_ids']) > entity_linker.max_input_tokens):
                    able_to_tokenize = False
                    break
            # in case of bad sentence tokenization and token overflow. 
            if not able_to_tokenize:
                with open(os.path.join(args.output_dir, "ERRORS", instance["id"]), "w", encoding="utf-8") as f:
                    json.dump(instance, f)
                failures += 1
                continue
            indices[instance["id"]] = (len(sentences), len(sentences) + len(sent_tokenized))
            sentences.extend(sent_tokenized)

        mentions = entity_linker.do_el(sentences)

        for key, value in indices.items():
            file_path = os.path.join(args.output_dir, key)
            start, end = value
            result = mentions[start: end]
            
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(result, f)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--signal_dir", help="Directory of Signal 1M Dataset", required=True)
    parser.add_argument("--seed", default=6556)
    parser.add_argument("--batch_size", default=128)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    main(args)