import nltk
from src.data.conll import ConLL2003

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
