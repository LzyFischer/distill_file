import re

# ----------------------------------------------------------------------
# 1) gold-label conversion
# ----------------------------------------------------------------------

_GOLD_MAP = {
    "entailment":    "true",
    "neutral":       "neither",
    "contradiction": "false",
}

def convert_gold(label: str) -> str:
    """
    Convert an NLI gold label to the {true, neither, false} scheme.
    
    >>> convert_gold("entailment")     # 'true'
    >>> convert_gold("Neutral")        # 'neither'
    >>> convert_gold("CONTRADICTION")  # 'false'
    """
    try:
        return _GOLD_MAP[label.strip().lower()]
    except KeyError as e:
        raise ValueError(f"Unknown gold label: {label!r}") from e


# ----------------------------------------------------------------------
# 2) answer extraction  (True / False / Neither)
# ----------------------------------------------------------------------

import re

_ANS_RE = re.compile(
    r"""
    answer            # literal “answer”  (case-insensitive)
    \s*[:\-]?\s*      # optional colon or dash
    ['"`\(]*\s*       # optional opening quote / parenthesis
    (?P<ans>true|false|neither)   # the token we want
    \s*['"`\)]*       # optional closing punctuation
    \s*\.?\s*         # optional period **and** trailing spaces
    """,
    re.IGNORECASE | re.VERBOSE,
)

def extract_answer(text: str) -> str:
    m = _ANS_RE.search(text)
    return m.group("ans").lower() if m else "N/A"

def extract_answer(text: str) -> str:
    """
    Return 'true', 'false', or 'neither' if found; else 'N/A'.
    
    >>> s = '''... Answer: True.  \nExplanation: ...'''
    >>> extract_answer(s)  # 'true'
    """
    m = _ANS_RE.search(text)
    return m.group("ans").lower() if m else "N/A"


# ----------------------------------------------------------------------
# quick demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    gold_demo = ["entailment", "neutral", "contradiction"]
    print("Gold → converted:", [convert_gold(g) for g in gold_demo])
    text_demo = '
    print("Extracted answer:", extract_answer(text_demo))