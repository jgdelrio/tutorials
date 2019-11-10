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

    replacements_set1 = [(r"i'm ", "i am "), (r"he's ", 'he is '), (r"she's ", 'she is '),
                         (r"it's ", 'it is '), (r"\'s", ' '), (r"\'ve ", " have "),
                         (r"d\''", ''), (r"can't ", "cannot "), (r"what's ", "what is "),
                         ("\n", " "), (" x ", " "), (" e ", " "), (" e.", " ")]

    replacements_set2 = [("u.s.", "united states"), (" ec ", " european community "),
                         (r"£", "pound"), (r"€", "euro"), (" dlrs", " dollars"),
                         (" mln", " million"), (" pct", " percentage")]

    replacements_set3 = [("1st ", "first "), ("2nd ", "second "), ("3rd ", "third "),
                         ("qtr", "quarter"), ("cts", "cents"), ("shr", "share"),
                         (" vs ", " versus "), (" inc ", " incorporated "), (" ltd ", " limited "),
                         (" &lt", " limited"), ("wk", "week"), ("prev ", "previous ")]

    replacements = [*replacements_set1, *replacements_set2, *replacements_set3]

    for rpl in replacements:
        text = text.replace(rpl[0], rpl[1])

    subs_set = [(r"[^A-Za-z0-9^,!.\/'+-=]", " "), (r"\'", " "), (r"@", " "), (r" n't ", " not "),
                (r"\'re", " are "), (r"\'d", " would"), (r"let\'s", "let us"),
                (r"\'ll", " will "), (r"y\'all", "you all"), (r" doin\' ", " doing "),
                (r"(\d+)(k)", r"\g<1>000")]

    for sub in subs_set:
        text = re.sub(sub[0], sub[1], text)

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
