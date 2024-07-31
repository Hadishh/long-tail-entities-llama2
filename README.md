# long-tail-entities-llama2
This project is a study on the long-tail entity detection using modern tools, including Large Language Models (LLM). \[ [Full Report](https://github.com/Hadishh/long-tail-entities-llama2/blob/main/long-tail-entities.pdf) \]

## Reproduce Results
### 1. Installing requirements
```
pip install -r requirements.txt
``` 
### 2. Run Universal NER 
Run the following command to see the exact arguments for unversal NER.
```
python -m src.tasks.universal_ner --help
```
### 3. Run SpEL
First of all, run the "download_spel_checkpoints.sh" bash file to download the checkpoints of SpEL model. You can config [base_model.cfg](https://github.com/Hadishh/long-tail-entities-llama2/blob/main/src/spel/base_model.cfg) to either use large or base model.

Your SpEL is ready to run. Execute the following command to see the exact arguments of the spel module. 
```
python -m src.tasks.spel --help
```

