import base64
from typing import Any, Dict


def parse_image_response(sdk_response: Any) -> Dict[str, list]:
    """
    Converts a Gemini image SDK response into a dictionary with 'text' and 'images' keys.

    Args:
        sdk_response: The Gemini SDK HTTP response.

    Returns:
        dict: {'text': str, 'images': [base64 strings]}
    """
    result = {"text": "", "images": []}

    candidate = sdk_response.candidates[0]
    for part in candidate.content.parts:
        # Extract text safely
        text = getattr(part, "text", "")
        if text:
            result["text"] += text.strip()
        # Extract inline_data safely
        inline_data = getattr(part, "inline_data", None)
        if inline_data and getattr(inline_data, "data", None):
            mime_type = getattr(inline_data, "mime_type", "")
            if mime_type.startswith("image/"):
                img_bytes = inline_data.data
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                result["images"].append(img_b64)
    return result
