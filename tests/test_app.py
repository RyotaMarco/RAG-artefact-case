"""Check chat text rendering helpers without starting Streamlit."""

from app import escape_currency


def test_currency_amounts_are_never_rendered_as_latex() -> None:
    """Two R$ on one line would otherwise turn the text between them into a formula."""
    line = "Cartão de crédito: R$ 1.199,00 em 10x de R$ 119,90"
    assert escape_currency(line) == "Cartão de crédito: R\\$ 1.199,00 em 10x de R\\$ 119,90"
    assert escape_currency("Sem valores.") == "Sem valores."
