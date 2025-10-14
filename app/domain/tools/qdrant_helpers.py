import re


def extract_qdrant_texts(
    results: list,
    min_score: float = 0.0,
    limit: int = None,
    sort_desc: bool = True,
    sanitize_text: bool = True,
) -> list[str]:
    filtered = [r for r in results if hasattr(r, "score") and r.score >= min_score]
    if sort_desc:
        filtered.sort(key=lambda r: r.score, reverse=True)
    if limit is not None:
        filtered = filtered[:limit]
    output = []
    for r in filtered:
        if hasattr(r, "payload") and "text" in r.payload:
            val = r.payload.get("text", "")
            output.append(sanitize_qdrant_text(val) if sanitize_text else val)
    return output


def sanitize_qdrant_text(text: str) -> str:
    """
    Sanitize Qdrant extracted content by removing/normalizing unwanted characters.
    - Replaces newlines, tabs, and non-breaking spaces with spaces.
    - Collapses multiple spaces into one.
    - Trims leading/trailing whitespace.
    """
    if not isinstance(text, str):
        return ""
    cleaned = text.replace("\n", " ").replace("\xa0", " ").replace("\t", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()
