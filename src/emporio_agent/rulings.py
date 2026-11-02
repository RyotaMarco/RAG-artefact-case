"""Business rulings for points the manual leaves open, loaded from data/store_rulings.yaml."""

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

RULINGS_FILE = "store_rulings.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ShippingRulings(_Section):
    """How the manual's free-shipping threshold is applied."""

    free_at_exact_threshold: bool
    threshold_on_final_amount: bool


class ReturnRulings(_Section):
    """When preference exchanges start and how in-store exchanges are handled."""

    preference_counts_from: Literal["receipt", "purchase"]
    in_store_preference: Literal["whatsapp", "window"]


class CombinedPaymentRulings(_Section):
    """Which payment methods may be combined in one order."""

    methods: list[Literal["pix", "credit", "debit", "boleto"]] = Field(min_length=2)


class ComplaintRulings(_Section):
    """Limits and data kept for complaint records."""

    max_per_conversation: int = Field(ge=1, le=20)
    keep_order_email: bool


class StoreRulings(_Section):
    """Every business ruling the deterministic tools depend on."""

    shipping: ShippingRulings
    returns: ReturnRulings
    combined_payment: CombinedPaymentRulings
    complaints: ComplaintRulings


def load_rulings(data_dir: Path) -> tuple[StoreRulings | None, str]:
    """Read and validate the rulings file, returning it with a content fingerprint.

    A missing file yields None, which disables the rules that depend on it.

    Raises:
        ValueError: The file exists but is not valid YAML or does not match the schema.
    """
    path = data_dir / RULINGS_FILE
    if not path.is_file():
        return None, "no-rulings"
    content = path.read_bytes()
    try:
        raw: object = yaml.safe_load(content)
        rulings = StoreRulings.model_validate(raw)
    except (yaml.YAMLError, ValidationError) as error:
        raise ValueError(f"{RULINGS_FILE} is invalid: {error}") from error
    return rulings, hashlib.sha256(content).hexdigest()
