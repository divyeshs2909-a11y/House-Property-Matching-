import re

_REPL = {
    "â€”": "—",
    "â€“": "–",
    "â€™": "'",
    "â€œ": '"',
    "â€�": '"',
    "â€¦": "...",
    "Â ": " ",
    "\u00a0": " ",
}

def fix_mojibake(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    for k, v in _REPL.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s).strip()
    return s
