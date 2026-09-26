def wrap(text, width):
    """Greedy word wrap. Lines never exceed width, except a single word longer than width."""
    if width < 1:
        raise ValueError("width must be positive")
    lines = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def truncate(text, width, ellipsis="…"):
    """Shorten text to at most width characters, ending with ellipsis when cut."""
    if len(text) <= width:
        return text
    if width <= len(ellipsis):
        return ellipsis[:width]
    return text[: width - len(ellipsis)] + ellipsis
