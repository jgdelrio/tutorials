import re
import unicodedata

import nltk
from nltk.tokenize import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import spacy
from textblob import TextBlob


SPECIAL_CHARS = "`1234567890-=~@#$%^&*()_+[!{;”:\’'><.,/?”}]"

stop_word_list = stopwords.words('english')
token = ToktokTokenizer()
lemma = WordNetLemmatizer()

# Dictionary for the transformation from wordnet tags to lemmatize characters
tag_dict_wordnet = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

tag_dict_textblob = {"J": 'a',
                      "N": 'n',
                      "V": 'v',
                      "R": 'r'}


def text_replacements(text):
    text = text.lower()
    text = text.replace(r"i'm ", "i am ")
    text = text.replace(r"he's ", 'he is ')
    text = text.replace(r"she's ", 'she is ')
    text = text.replace(r"it's ", 'it is ')
    text = text.replace(r"\'s", ' ')
    text = text.replace(r"d\''", '')
    text = text.replace(r"\'ve ", " have ")
    text = text.replace(r"can't ", "cannot ")
    text = text.replace(r"what's ", "what is ")
    text = text.replace(r"£", "pound")
    text = text.replace(r"€", "euro")
    text = text.replace("\n", " ")

    text = text.replace("u.s.", "united states")
    text = text.replace(" ec ", " european community ")
    text = text.replace(" dlrs", " dollars")
    text = text.replace(" mln", " million")
    text = text.replace(" pct", " percentage")
    text = text.replace(" x ", " ")
    text = text.replace(" e ", " ")
    text = text.replace(" e.", " ")
    text = text.replace("1st ", "first ")
    text = text.replace("2nd ", "second ")
    text = text.replace("qtr", "quarter")
    text = text.replace("cts", "cents")
    text = text.replace("shr", "share")
    text = text.replace(" vs ", " versus ")
    text = text.replace(" inc ", " incorporated ")
    text = text.replace(" ltd ", " limited ")
    text = text.replace(" &lt", " limited")
    text = text.replace("wk", "week")
    text = text.replace("prev ", "previous ")

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"\'", " ", text)
    text = re.sub(r"@", " ", text)
    text = re.sub(r" n't ", " not ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"let\'s", "let us", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"y\'all", "you all", text)
    text = re.sub(r" doin\' ", " doing ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    # Extend with other cleaning text operations...
    return text


def remove_accents(text):
    return unicodedata.normalize(
        'NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def remove_special(text):
    for w in text:
        if w in SPECIAL_CHARS:
            text = text.replace(w, ' ')
    return text


def clean_text(text):
    text = remove_special(remove_accents(text_replacements(text)))
    return text


def get_wordnet_pos(word):
    """Map POS tag to first character for lemmatize()"""
    tag = nltk.pos_tag([word])[0][1][0].upper()

    return tag_dict_wordnet.get(tag, wordnet.NOUN)


def get_textblob_pos(word):
    """Map POS tag to first character for lemmatize()"""
    tag = nltk.pos_tag([word])[0][1][0].upper()

    return tag_dict_textblob.get(tag, wordnet.NOUN)


def remove_stop_words_and_lemmatize(text, token_output=False):
    word_list = [x.strip() for x in token.tokenize(text)]
    output_list = [lemma.lemmatize(x, get_wordnet_pos(x)) for x in word_list if not x in stop_word_list]

    if token_output:
        return output_list
    else:
        return " ".join(output_list)


def lemmatize_doc(text, lemmatizer='nltk'):
    if lemmatizer == 'nltk':
        lematized_doc = [lemma.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)]
        return ' '.join(lematized_doc)

    elif lemmatizer == 'spacy':
        nlp = spacy.load('en', disable=['parser', 'ner'])
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc])

    elif lemmatizer == 'textblob':
        sentence = TextBlob(text)
        words_and_tags = [(w, tag_dict_textblob.get(pos[0], 'n')) for w, pos in sentence.tags]
        lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
        return " ".join(lemmatized_list)

