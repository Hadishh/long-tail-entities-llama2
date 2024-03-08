import argparse
from tqdm import tqdm
import nltk
import os
import json

import sentence_transformers
from sentence_transformers import util
from src.data.conll import ConLL2003
from src.data.signal_dataset import SignalDataset


def main(args):
    device = "cuda:0"
    NER_TYPES = ["PER", "ORG", "LOC", "MISC"]
    os.makedirs(args.output_dir, exist_ok=True)
    embedder = sentence_transformers.SentenceTransformer(args.model)

    print("Loading ConLL 2003 dataset")
    corpuses = { t: ConLL2003(t) for t in NER_TYPES}
    corpus_embeddings = dict()

    for ner_type in NER_TYPES:
        print(f"Encoding Corpus for {ner_type}")
        corpus = [d["sentence"] for d in corpuses[ner_type]]
        embeddings = embedder.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
        embeddings = embeddings.to(device)
        corpus_embeddings[ner_type] = embeddings


    signal_ds = SignalDataset(args.signal_dir)
    print("Processing Signal 1M Dataset:")
    for i in tqdm(range(len(signal_ds))):
        signal_d = signal_ds[i]

        sentences = nltk.sent_tokenize(signal_d["content"])
        id = signal_d["id"]

        json_data = {"id": id}

        query_embeddings = embedder.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
        for ner_type in NER_TYPES:
            #list of hits for each sentence in the query. Reference: https://www.sbert.net/examples/applications/semantic-search/README.html
            hits = util.semantic_search(query_embeddings, corpus_embeddings[ner_type], top_k=args.topk)
            conll_ids = list()
            # Extract conll ids related to each sentence
            for sent_hits in hits:
                conll_ids.append(list())
                for hit in sent_hits:
                    conll_id = corpuses[ner_type][hit["corpus_id"]]["id"]
                    conll_ids[-1].append(conll_id)
            
            json_data[ner_type] = conll_ids
        
        out_path = os.path.join(args.output_dir, id)
        with open(out_path, "w") as f:
            json.dump(json_data, f)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--signal_dir", help="Directory of Signal 1M Dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--topk", default=50, help="Top K similar sentences in ConLL2003", type=int)
    parser.add_argument("--model", default="all-mpnet-base-v2", help="Sentence transformer used for semantic search")

    args = parser.parse_args()

    main(args)
