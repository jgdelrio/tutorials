import re


def words_abbreviation(words, split_by=r"[ \-,;:]"):
    """Build some abbreviations from a list or words"""
    if not isinstance(words, str):
        raise TypeError("The parameter 'word' must be a string")
    split_regex = re.compile(split_by)
    splitted = split_regex.split(words)
    if len(splitted) == 1:
        return splitted[0][:2].upper()
    else:
        return "".join([k[0].upper() for k in splitted])


def list_abbreviations(word_list, split_by=None):
    if not isinstance(word_list, list):
        raise TypeError("The parameter 'word_list' must be a list")
    if split_by:
        return [words_abbreviation(m, split_by=split_by) for m in word_list]
    return [words_abbreviation(m) for m in word_list]