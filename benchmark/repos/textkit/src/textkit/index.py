from collections import defaultdict

from textkit.tokens import tokenize


class Index:
    """Inverted index from word to the ids of the documents containing it.

    Lookups are case-insensitive.
    """

    def __init__(self):
        self._postings = defaultdict(set)

    def add(self, doc_id, text):
        for word, _start, _end in tokenize(text):
            self._postings[word].add(doc_id)

    def lookup(self, term):
        return sorted(self._postings.get(term.casefold(), set()))
