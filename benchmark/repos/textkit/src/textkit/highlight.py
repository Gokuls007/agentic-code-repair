from textkit.tokens import find_spans


def highlight(text, term, marker=("[", "]")):
    """Wrap every whole-word, case-insensitive occurrence of term in markers."""
    out = []
    pos = 0
    for start, end in find_spans(text, term):
        out.append(text[pos:start])
        out.append(marker[0] + text[start:end] + marker[1])
        pos = end
    out.append(text[pos:])
    return "".join(out)
