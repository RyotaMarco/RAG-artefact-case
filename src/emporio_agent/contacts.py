"""Contact formats shared by internal-contact masking and complaint redaction."""

import re

# Brazilian phones with or without +55, parentheses, spaces, dots or a hyphen:
# (67) 3341-4444, 67 3341 4444, +55 67 99876.5432, (67)33414444.
PHONE = re.compile(r"(?:\+?55[\s.-]*)?(?:\(\d{2}\)|\b\d{2})[\s.-]*9?\d{4}[\s.-]?\d{4}\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
CPF = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")


def phone_digits(text: str) -> str:
    """Return a phone's area code and number, without the +55 country code."""
    digits = re.sub(r"\D", "", text)
    return digits[2:] if digits.startswith("55") and len(digits) in {12, 13} else digits
