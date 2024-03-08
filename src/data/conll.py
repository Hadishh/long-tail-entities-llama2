from datasets import load_dataset
import json

class ConLL2003():
    
    def __tag_sentence(self, tokens, occurs):
        tokens = tokens.copy()
        for s, e in occurs:
            tokens[s] = "@@" + tokens[s]
            tokens[e] = tokens[e] + "##"
        
        return " ".join(tokens)
    
    
    def __process_entry(self, entry):
        ner_type = self.ner_type
        tags = entry["ner_tags"]
        B_TAG_IDX = self.tag2idx[f"B-{ner_type}"]
        I_TAG_IDX = self.tag2idx[f"I-{ner_type}"]
        occurances = []
        start_idx = -1
        end_idx = -1

        for i in range(len(tags)):
            if tags[i] == B_TAG_IDX:
                if start_idx != -1:
                    occurances.append((start_idx, i - 1))
                start_idx = i
                end_idx = -1
            elif tags[i] == I_TAG_IDX and start_idx != -1:
                end_idx = i - 1
            elif tags[i] == I_TAG_IDX and start_idx == -1:
                start_idx = i
            elif start_idx != -1:
                end_idx = i - 1
                occurances.append((start_idx, end_idx))
                start_idx = -1
                end_idx = -1
        
        if start_idx != -1:
            end_idx = len(tags) - 1
            occurances.append((start_idx, end_idx))
            start_idx = -1
            end_idx = -1
        if occurances:
            self.data.append({"id": entry["id"], 
                                        "tokens": entry["tokens"], 
                                        "sentence": " ".join(entry["tokens"]),
                                        "tagged": self.__tag_sentence(entry["tokens"], occurances),
                                        "occurs": occurances}
                                    )

    def __init__(self, ner_type):
        
        # Tags are gathered from https://huggingface.co/datasets/conll2003
        self.tag2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

        self.ner_type = ner_type.upper()
        hf_data = load_dataset("conll2003", revision="refs/convert/parquet")
        self.data = []
        
        for split in hf_data:
            for entry in hf_data[split]:
                self.__process_entry(entry)
        
        self.id2idx = {d["id"] : i for i, d in enumerate(self.data)}
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

    def get_by_id(self, id):
        return self.data[self.id2idx[id]]
