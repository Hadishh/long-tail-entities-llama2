import nltk
from src.data.conll import ConLL2003
from src.conversation import get_conv_template


SELF_VERIFICATION_INITIAL_TEMPLATE = \
(
    "I am an excellent linguist. The task is to verify whether the word is a {} entity extracted from the given sentence.\n"

)

SELF_VERIFICATION_EXAMPLE_TEMPLATE = \
(
    "The given sentence:{}\n"
    "Is the word \"{}\" in the given sentence a {} entity? Please answer with Yes or No.\n"
    "{}.\n"
    "\n\n"
)

NER_INITIAL_TEMPLATE = \
(
    "I am an excellent linguist. The task is to label {} entities in the given sentence. Below are some examples:\n"
)

NER_EXAMPLE_TEMPLATE = \
(
    "Input: {}\n"
    "Output: {}\n"
    "\n\n"
)


NER_TYPES = ["PER", "ORG", "LOC", "MISC"]

def sentence_tokenize(article):
    return nltk.sent_tokenize(article)

def load_conll_corpuses():
    corpuses = { t: ConLL2003(t) for t in NER_TYPES}
    return corpuses


def preprocess_instance(source):
    """Obtained from https://github.com/universal-ner/universal-ner/blob/main/src/utils.py"""
    conv = get_conv_template("ie_as_qa")
    for j, sentence in enumerate(source):
        value = sentence['value']
        if j == len(source) - 1:
            value = None
        conv.append_message(conv.roles[j % 2], value)
    prompt = conv.get_prompt()
    return prompt

def get_response(responses):
    """Obtained from https://github.com/universal-ner/universal-ner/blob/main/src/utils.py"""
    responses = [r.split('ASSISTANT:')[-1].strip() for r in responses]
    return responses
