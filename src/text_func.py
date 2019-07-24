import re
import unicodedata


SPECIAL_CHARS = "`1234567890-=~@#$%^&*()_+[!{;”:\’'><.,/?”}]"


def clean_text(text):
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

