"""Check the business rulings file and how each ruling changes the deterministic tools."""

from dataclasses import replace
from pathlib import Path

import pytest
from agent_fixtures import manual_snapshot, small_snapshot

from emporio_agent.agent import ConversationAccess, StoreTools, conversation_token
from emporio_agent.rulings import RULINGS_FILE, StoreRulings, load_rulings
from emporio_agent.runtime import Snapshot

DATA = Path(__file__).parents[1] / "data"


@pytest.fixture(scope="module")
def manual() -> Snapshot:
    """Load the delivered sources and rulings once."""
    return manual_snapshot()


def tools(snapshot: Snapshot, tmp_path: Path) -> StoreTools:
    """Bind tools to a private conversation without starting a model."""
    access = ConversationAccess(tmp_path / "sessions.sqlite3", conversation_token(), "generation")
    return StoreTools(snapshot, access)


def with_rulings(manual: Snapshot, **changes: dict[str, object]) -> Snapshot:
    """Return the snapshot with some ruling sections changed."""
    assert manual.rulings is not None
    sections = {
        name: section.model_copy(update=changes[name]) if name in changes else section
        for name, section in (
            ("shipping", manual.rulings.shipping),
            ("returns", manual.rulings.returns),
            ("combined_payment", manual.rulings.combined_payment),
            ("complaints", manual.rulings.complaints),
        )
    }
    return replace(manual, rulings=StoreRulings.model_validate(sections))


def test_delivered_rulings_file_holds_the_agreed_business_decisions() -> None:
    """The repository file loads and states every decision agreed with the store."""
    rulings, fingerprint = load_rulings(DATA)
    assert rulings is not None and len(fingerprint) == 64
    assert rulings.shipping.free_at_exact_threshold is False
    assert rulings.shipping.threshold_on_final_amount is True
    assert rulings.returns.preference_counts_from == "receipt"
    assert rulings.returns.in_store_preference == "whatsapp"
    assert rulings.combined_payment.methods == ["pix", "credit", "debit"]
    assert rulings.complaints.max_per_conversation == 3
    assert rulings.complaints.keep_order_email is True


def test_missing_rulings_file_disables_dependent_rules(tmp_path: Path) -> None:
    """Without the file nothing is assumed."""
    assert load_rulings(tmp_path) == (None, "no-rulings")


@pytest.mark.parametrize(
    "content",
    [
        "shipping: [",
        "shipping: {}",
        (DATA / RULINGS_FILE)
        .read_text(encoding="utf-8")
        .replace("free_at_exact_threshold: false", "free_at_exact_threshold: 'no'"),
        (DATA / RULINGS_FILE).read_text(encoding="utf-8") + "\nextra: 1\n",
        (DATA / RULINGS_FILE).read_text(encoding="utf-8").replace("receipt", "tomorrow", 1),
    ],
)
def test_invalid_rulings_are_reported(tmp_path: Path, content: str) -> None:
    """Broken YAML, missing keys, strings for booleans and unknown keys are rejected."""
    (tmp_path / RULINGS_FILE).write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="store_rulings.yaml is invalid"):
        load_rulings(tmp_path)


def test_threshold_basis_ruling_changes_the_fee_for_pix(manual: Snapshot, tmp_path: Path) -> None:
    """Shelby SU-21S + Kala KA-C: R$ 509,80 in catalog, R$ 484,31 paid by PIX."""
    order = [{"product_id": 119, "quantity": 1}, {"product_id": 120, "quantity": 1}]
    final = tools(manual, tmp_path).calculate_quote(order)
    catalog = tools(
        with_rulings(manual, shipping={"threshold_on_final_amount": False}), tmp_path / "c"
    ).calculate_quote(order)
    assert final["order_catalog_total"] == "R$ 509,80"
    assert final["total"] == catalog["total"] == "R$ 484,31"
    assert final["metro_region_shipping"] == "R$ 35,00"
    assert catalog["metro_region_shipping"] == "R$ 0,00"


@pytest.mark.parametrize(("free_at_threshold", "fee"), [(False, "R$ 35,00"), (True, "R$ 0,00")])
def test_exact_threshold_ruling(
    manual: Snapshot, tmp_path: Path, free_at_threshold: bool, fee: str
) -> None:
    """An order of exactly R$ 500,00 pays the fee unless the store rules otherwise."""
    ruled = with_rulings(manual, shipping={"free_at_exact_threshold": free_at_threshold})
    snapshot = replace(
        small_snapshot(price_cents=25001),
        policies=manual.policies,
        rules=manual.rules,
        rulings=ruled.rulings,
    )
    result = tools(snapshot, tmp_path).calculate_quote(
        [{"product_id": 101}, {"product_id": 202}], payment_method="credit"
    )
    assert result["total"] == "R$ 500,00"
    assert result["metro_region_shipping"] == fee


def test_in_store_preference_ruling_switches_between_referral_and_window(
    manual: Snapshot, tmp_path: Path
) -> None:
    """'whatsapp' refers to staff; 'window' applies the manual's days from the purchase."""
    store = tools(manual, tmp_path)
    referral = store.check_return_window("preference", "store", "2026-11-01")
    assert referral["applicable"] is False
    window = tools(
        with_rulings(manual, returns={"in_store_preference": "window"}), tmp_path / "w"
    ).check_return_window("preference", "store", "2026-11-01")
    assert window["deadline"] == "2026-11-08"


def test_preference_can_count_from_the_purchase(manual: Snapshot, tmp_path: Path) -> None:
    """With 'purchase', an online exchange counts from the purchase date."""
    store = tools(with_rulings(manual, returns={"preference_counts_from": "purchase"}), tmp_path)
    result = store.check_return_window("preference", "online", "2026-10-30", "2026-11-01")
    assert result["deadline"] == "2026-11-06"


def test_combined_methods_follow_the_rulings(manual: Snapshot, tmp_path: Path) -> None:
    """Boleto is refused by default and accepted only when the store allows it."""
    order = [{"product_id": 94, "quantity": 1}]
    default = tools(manual, tmp_path).calculate_split_payment(order, "boleto", "1000", "pix")
    assert "Boleto takes up to 3 business days" in str(default["error"])
    allowed = with_rulings(
        manual, combined_payment={"methods": ["pix", "credit", "debit", "boleto"]}
    )
    result = tools(allowed, tmp_path / "b").calculate_split_payment(order, "boleto", "1000", "pix")
    assert result["total"] == "R$ 5.519,08"


def test_tools_fail_closed_without_rulings(manual: Snapshot, tmp_path: Path) -> None:
    """Without the rulings file, shipping, split payment and complaints refer to staff."""
    store = tools(replace(manual, rulings=None), tmp_path)
    assert "not confirmed" in str(store.calculate_quote([{"product_id": 89}])["shipping_note"])
    split = store.calculate_split_payment([{"product_id": 94}], "pix", "1000", "credit")
    assert "rulings are unavailable" in str(split["error"])
    assert "does not confirm" in str(store.check_return_window("preference", "online")["error"])
    assert "regret" in str(store.check_return_window("regret", "store")["reason"])
