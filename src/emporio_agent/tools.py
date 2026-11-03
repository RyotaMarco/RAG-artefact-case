"""The agent's tools, each bound to one source snapshot and one private order scope."""

from datetime import datetime
from decimal import Decimal, localcontext
from zoneinfo import ZoneInfo

from emporio_agent.access import ConversationAccess
from emporio_agent.complaints import record_complaint
from emporio_agent.data import Row
from emporio_agent.documents import Citation
from emporio_agent.money import (
    check_cents,
    discounted_cents,
    format_brl,
    parse_customer_amount,
    split_installments,
)
from emporio_agent.policies import PolicyUnavailableError, normalize_method
from emporio_agent.runtime import Snapshot
from emporio_agent.sql import QueryEngine, QueryError
from emporio_agent.store_rules import (
    RETURN_REASONS,
    ReturnTerms,
    metro_shipping_cents,
    opening_status,
    return_window,
)


def _citations(values: tuple[Citation, ...]) -> list[dict[str, object]]:
    """Return portable source references without leaking absolute filesystem paths."""
    return [
        {"file": value.source, "pages": value.pages, "section": value.section} for value in values
    ]


class StoreTools:
    """Bind each tool call to one source snapshot and one private order scope."""

    def __init__(
        self,
        snapshot: Snapshot,
        access: ConversationAccess,
        timezone: str = "America/Campo_Grande",
    ) -> None:
        """Retain references and query failures only within the current agent run."""
        self.snapshot = snapshot
        self.access = access
        self.timezone = ZoneInfo(timezone)
        self.engine = QueryEngine(snapshot.catalog)
        self.evidence = snapshot.evidence
        self.sources: list[dict[str, object]] = []
        self.query_failures = 0

    def search_policies(self, query: str) -> dict[str, object]:
        """Search active store PDFs using hybrid retrieval.

        Parameters:
            query: A policy question, preferably using Portuguese search terms.

        Returns:
            Relevant passages with their source pages and sections, or an explicit error.
        """
        if self.snapshot.policies is None:
            return {"error": "No policy PDF was supplied. This capability is unavailable."}
        try:
            hits = self.snapshot.policies.search(query, limit=5)
            passages: list[dict[str, object]] = []
            for hit in hits:
                references = _citations(hit.citations)
                self.sources.extend(references)
                passages.append({"text": self.evidence.mask(hit.text), "sources": references})
            return {"passages": passages, "generation": self.snapshot.generation_id}
        except ValueError as error:
            return {"error": f"Policy retrieval is unavailable: {error}"}

    def now(self) -> datetime:
        """Return the current store-local time; tests replace this clock."""
        return datetime.now(self.timezone)

    def store_open_now(self) -> dict[str, object]:
        """Check the regular opening hours against the current store-local time.

        Returns:
            Whether the store is open by regular hours, today's hours, the next opening and a
            holiday warning, or an error when the manual does not confirm the hours.
        """
        if not self.evidence.has("hours"):
            return {
                "error": "The active manual does not confirm opening hours. Refer the customer "
                "to staff."
            }
        self.sources.extend(_citations(self.evidence.citations("hours")))
        status = opening_status(self.now(), self.evidence.values.weekly_hours or {})
        status["whatsapp"] = self.evidence.whatsapp
        return status

    def check_return_window(
        self,
        reason: str,
        channel: str,
        purchased_on: str | None = None,
        received_on: str | None = None,
    ) -> dict[str, object]:
        """Check return, exchange or defect deadlines from dates the customer states.

        Parameters:
            reason: regret (withdrawal), preference (exchange) or defect.
            channel: online or store, as stated by the customer.
            purchased_on: Purchase date YYYY-MM-DD, if the customer gave it. When omitted for
                a verified order, the order date from the store's records is used.
            received_on: Receipt date YYYY-MM-DD, only if the customer gave it. Never use
                order_date or estimated_delivery from the order records.

        Returns:
            Deadlines and whether each window is still open, or what to ask for. Not an approval.
        """
        terms = self._return_terms()
        if (
            reason not in RETURN_REASONS
            or not self.evidence.has(reason)
            or (reason == "preference" and self.snapshot.rulings is None)
        ):
            return {
                "error": "The active manual does not confirm this rule. Refer the customer to "
                "staff."
            }
        # The order record is the purchase date; it never stands in for the receipt date.
        # Only defects, in-store purchases and purchase-based preference windows count from
        # the purchase; an online withdrawal never needs the order record.
        needs_purchase = (
            reason == "defect"
            or channel == "store"
            or (reason == "preference" and terms.preference_counts_from == "purchase")
        )
        source = "customer" if purchased_on else None
        verified = self.access.verified_order()
        if not purchased_on and needs_purchase and verified is not None:
            purchased_on = self.engine.order_date(verified)
            source = "order record" if purchased_on else None
        try:
            result = return_window(
                reason, channel, purchased_on, received_on, self.now().date(), terms
            )
        except ValueError as error:
            return {"error": str(error)}
        if source is not None:
            result["purchase_date_used"] = purchased_on
            result["purchase_date_source"] = source
        if source == "order record":
            result["confirm_with_customer"] = (
                f"The order record shows a purchase date of {purchased_on}. Tell the customer the "
                "record shows it and ask them to confirm; do not state it as a fact."
            )
        keys = (reason, "not_exchangeable") if reason != "defect" else (reason,)
        self.sources.extend(_citations(self.evidence.citations(*keys)))
        result["note"] = (
            f"Informational only, not an approval. Staff confirm on WhatsApp {self.evidence.whatsapp}."
        )
        return result

    def _return_terms(self) -> ReturnTerms:
        """Combine manual deadlines with the store's exchange rulings.

        Missing values default to zero or empty; has(reason) blocks any rule that needs them.
        """
        values, rulings = self.evidence.values, self.snapshot.rulings
        return ReturnTerms(
            regret_days=values.regret_days or 0,
            refund_business_days=values.refund_business_days or 0,
            preference_days=values.preference_days or 0,
            defect_days=values.defect_days or 0,
            manufacturer_warranty=values.manufacturer_warranty or "",
            preference_counts_from=rulings.returns.preference_counts_from if rulings else "",
            in_store_preference=rulings.returns.in_store_preference if rulings else "",
        )

    def register_complaint(self, summary: str) -> dict[str, object]:
        """Record a complaint about the verified order for staff follow-up.

        Parameters:
            summary: A short description of the problem, without personal data.

        Returns:
            The protocol, the response time and the WhatsApp message to send, or an error.
        """
        rulings = self.snapshot.rulings
        if not self.evidence.has("complaints") or self.evidence.whatsapp is None or rulings is None:
            return {
                "error": "The active manual does not confirm the complaint process. Refer the "
                "customer to staff."
            }
        order_id = self.access.verified_order()
        if order_id is None:
            return {
                "error": "No verified order. Ask for the order number and email and call "
                "verify_order first. Without an order, refer the customer to WhatsApp "
                f"{self.evidence.whatsapp} without a protocol."
            }
        try:
            protocol = record_complaint(
                self.access.database,
                self.access.identity,
                order_id,
                summary,
                self.now(),
                limit=rulings.complaints.max_per_conversation,
                email=self.engine.order_email(order_id)
                if rulings.complaints.keep_order_email
                else None,
            )
        except ValueError as error:
            return {"error": str(error)}
        self.sources.extend(_citations(self.evidence.citations("complaints")))
        return {
            "protocol": protocol,
            "order_id": order_id,
            "response_time": f"Up to {self.evidence.values.complaint_hours} business hours.",
            "whatsapp": self.evidence.whatsapp,
            "whatsapp_message": f"Olá! Tenho uma reclamação registrada com o protocolo {protocol}.",
        }

    def verify_order(self, order_id: int, email: str) -> dict[str, object]:
        """Verify one order/contact pair before private order queries.

        Parameters:
            order_id: The customer's order number.
            email: The email associated with that order.

        Returns:
            A verification status without listing customer information.
        """
        if not self.snapshot.capabilities()["orders"]:
            return {"error": "Order identification sources were not supplied."}
        if self.access.verify(self.engine, order_id, email):
            verified = self.access.verified_order()
            first_name = None if verified is None else self.engine.customer_first_name(verified)
            return {"verified": True, "order_id": order_id, "customer_first_name": first_name}
        return {"verified": False, "error": "The supplied order details could not be verified."}

    def query_data(self, sql: str) -> dict[str, object]:
        """Execute one scoped read query over the currently available business tables.

        Parameters:
            sql: A plain SQLite SELECT using the supplied schema and integer-cent money bounds.

        Returns:
            Bounded rows, exact formatted money, units and original CSV sources, or an error.
        """
        if self.query_failures >= 2:
            return {"error": "The query retry limit has been reached for this request."}
        try:
            result = self.engine.query(sql, verified_order_id=self.access.verified_order())
            self.sources.extend({"file": source} for source in result.source_paths)
            return result.as_dict()
        except QueryError as error:
            self.query_failures += 1
            return {"error": str(error)}

    def _order_lines(self, items: list[dict[str, int]]) -> list[tuple[Row, int]]:
        """Merge repeated products and check each one's status and stock for the whole order."""
        if not 1 <= len(items) <= 20:
            raise ValueError("Provide between 1 and 20 order items.")
        quantities: dict[int, int] = {}
        for item in items:
            product_id, quantity = item.get("product_id"), item.get("quantity", 1)
            if type(product_id) is not int or type(quantity) is not int or quantity < 1:
                raise ValueError("Each item needs an integer product_id and a positive quantity.")
            quantities[product_id] = quantities.get(product_id, 0) + quantity
        lines: list[tuple[Row, int]] = []
        for product_id, quantity in quantities.items():
            product = next(
                (
                    row
                    for row in self.snapshot.catalog.tables["products"].rows
                    if row["product_id"] == product_id
                ),
                None,
            )
            if product is None:
                raise ValueError(f"Product {product_id} is not present in the current catalog.")
            stock = product["stock_quantity"]
            if (
                str(product["status"]).casefold() != "active"
                or not isinstance(stock, int)
                or stock <= 0
            ):
                raise ValueError(f"{product['name']} is not currently available for purchase.")
            if quantity > stock:
                raise ValueError(f"The requested quantity of {product['name']} exceeds stock.")
            lines.append((product, quantity))
        return lines

    def _promotion_rate(self, product: Row) -> Decimal:
        rates = {
            Decimal(str(row["discount_percent"]))
            for row in self.snapshot.catalog.tables["promotions"].rows
            if row["product_id"] == product["product_id"] and row["is_active"] == 1
        }
        if len(rates) > 1:
            raise PolicyUnavailableError(
                f"Multiple active discounts apply to {product['name']}. Staff must confirm the "
                "offer."
            )
        return max(rates) if rates else Decimal(0)

    def _order_totals(
        self, items: list[dict[str, int]]
    ) -> tuple[list[dict[str, object]], int, int, int]:
        """Return line details, the original total and regular/promotional subtotals in cents."""
        tables = self.snapshot.catalog.tables
        if "products" not in tables or "promotions" not in tables:
            raise PolicyUnavailableError("Product and promotion sources are required for a quote.")
        lines: list[dict[str, object]] = []
        original_total = regular = promoted = 0
        for product, quantity in self._order_lines(items):
            original = check_cents(check_cents(product["price_cents"]) * quantity)
            rate = self._promotion_rate(product)
            current = discounted_cents(original, str(rate))
            original_total = check_cents(original_total + original)
            if rate:
                promoted = check_cents(promoted + current)
            else:
                regular = check_cents(regular + current)
            lines.append(
                {
                    "product": product["name"],
                    "quantity": quantity,
                    "original_total": format_brl(original),
                    "current_catalog_total": format_brl(current),
                    "active_promotion_percent": str(rate),
                }
            )
        self.sources.extend({"file": path} for path in tables["products"].source_paths)
        self.sources.extend({"file": path} for path in tables["promotions"].source_paths)
        return lines, original_total, regular, promoted

    def _metro_shipping(self, total: int, catalog_total: int) -> dict[str, object]:
        """Report the metropolitan fee from the manual's values and the store's rulings."""
        values, rulings = self.evidence.values, self.snapshot.rulings
        if (
            not self.evidence.has("metro_shipping")
            or rulings is None
            or values.metro_threshold_cents is None
            or values.metro_fee_cents is None
        ):
            return {
                "shipping_note": "Shipping rules are not confirmed by the active manual. Staff "
                "quote shipping."
            }
        on_final = rulings.shipping.threshold_on_final_amount
        fee = metro_shipping_cents(
            total if on_final else catalog_total,
            threshold_cents=values.metro_threshold_cents,
            fee_cents=values.metro_fee_cents,
            free_at_threshold=rulings.shipping.free_at_exact_threshold,
        )
        self.sources.extend(_citations(self.evidence.citations("metro_shipping")))
        return {
            "metro_region_shipping": format_brl(fee),
            "total_with_metro_shipping": format_brl(check_cents(total + fee)),
            "shipping_note": "Campo Grande metropolitan region only, based on the "
            f"{'final amount paid' if on_final else 'catalog total'} for the whole order. "
            "Other cities depend on the destination CEP; staff quote "
            f"it on WhatsApp {self.evidence.whatsapp}.",
        }

    def calculate_quote(
        self, items: list[dict[str, int]], payment_method: str = "pix", installments: int = 1
    ) -> dict[str, object]:
        """Calculate one sourced payment quote for the whole order with exact Decimal arithmetic.

        Parameters:
            items: Every product in the order as {"product_id": id, "quantity": n}. Quote all
                products together in one call; never add separate quotes.
            payment_method: PIX, credit, debit or boleto, when supported by the policy.
            installments: Requested number of installments for the order total.

        Returns:
            Per-item prices, the order total, exact installments, shipping on the order total and
            policy references, or an error.
        """
        try:
            lines, original_total, regular, promoted = self._order_totals(items)
            catalog_total = check_cents(regular + promoted)
            # The PIX discount never applies to promotional prices, so it covers only the
            # regular-price subtotal; any other method is quoted on the whole order.
            if normalize_method(payment_method) == "pix":
                parts = [part for part in ((regular, False), (promoted, True)) if part[0]]
                parts = parts or [(0, False)]
            else:
                parts = [(catalog_total, bool(promoted))]
            with localcontext() as context:
                context.prec = 120
                quotes = [
                    self.snapshot.rules.quote(
                        Decimal(amount) / 100,
                        payment_method=payment_method,
                        installments=installments,
                        promotional=flag,
                    )
                    for amount, flag in parts
                ]
                total = check_cents(sum(int(quote.total * 100) for quote in quotes))
            amounts = split_installments(total, installments)
            references = _citations(
                tuple(dict.fromkeys(c for quote in quotes for c in quote.citations))
            )
            self.sources.extend(references)
            shipping = self._metro_shipping(total, catalog_total)
            notes = " ".join(dict.fromkeys(quote.note for quote in quotes if quote.note))
            return {
                "items": lines,
                "order_original_total": format_brl(original_total),
                "order_catalog_total": format_brl(catalog_total),
                "payment_method": payment_method,
                "total": format_brl(total),
                "installment_amounts": [format_brl(value) for value in amounts],
                "sources": references,
                "policy_note": notes or None if installments == 1 else None,
                "note": "One quote for the whole order: discounts, installments and shipping use "
                "the order total. Installment amounts sum exactly to it. This is not a purchase.",
                **shipping,
            }
        except (ValueError, ArithmeticError) as error:
            return {"error": str(error)}

    def _split_part(
        self, method: str, amount: int, count: int
    ) -> tuple[dict[str, object], tuple[Citation, ...]]:
        """Validate one part of a combined payment against the manual's payment rules."""
        rules = self.snapshot.rules
        if (method in {"debit", "boleto"} and method not in rules.single_payment_methods) or (
            method == "pix" and rules.pix_rate is None
        ):
            raise PolicyUnavailableError(
                f"The active manual does not list {method} payments. Refer to staff."
            )
        citations: tuple[Citation, ...] = ()
        if method == "credit":
            with localcontext() as context:
                context.prec = 120
                quote = rules.quote(
                    Decimal(amount) / 100, payment_method="credit", installments=count
                )
            citations = quote.citations
        elif count != 1:
            raise ValueError(f"The {method} part is paid in a single payment.")
        payment: dict[str, object] = {
            "method": method,
            "amount": format_brl(amount),
            "installments": count,
            "installment_amounts": [
                format_brl(value) for value in split_installments(amount, count)
            ],
        }
        return payment, citations

    def calculate_split_payment(
        self,
        items: list[dict[str, int]],
        first_method: str,
        first_amount: str,
        second_method: str,
        first_installments: int = 1,
        second_installments: int = 1,
    ) -> dict[str, object]:
        """Split the whole order between two payment methods, as the customer chose.

        Parameters:
            items: Every product in the order as {"product_id": id, "quantity": n}.
            first_method: pix, credit or debit.
            first_amount: Amount the customer stated for the first method, as written, like
                "1.000" or "1.000,50".
            second_method: A different method among pix, credit and debit; it pays the rest.
            first_installments: Credit installments for the first part, otherwise 1.
            second_installments: Credit installments for the second part, otherwise 1.

        Returns:
            Each part with exact installments, the order total and shipping, or an error.
        """
        try:
            rules = self.snapshot.rules
            if rules.conflicts:
                raise PolicyUnavailableError(
                    "Conflicting source policies require staff confirmation."
                )
            if rules.mixed_payment_minimum is None or not self.evidence.has("mixed_payment"):
                raise PolicyUnavailableError(
                    "The active manual does not confirm combined payments. Refer to staff."
                )
            methods = [normalize_method(name) for name in (first_method, second_method)]
            if self.snapshot.rulings is None:
                raise PolicyUnavailableError(
                    "The store's combined-payment rulings are unavailable. Refer to staff."
                )
            allowed = self.snapshot.rulings.combined_payment.methods
            if not set(methods) <= set(allowed) or methods[0] == methods[1]:
                boleto = (
                    ""
                    if "boleto" in allowed
                    else (" Boleto takes up to 3 business days to clear and is not combined.")
                )
                raise ValueError(
                    f"Combine two different methods among {', '.join(allowed)}.{boleto}"
                )
            lines, original_total, regular, promoted = self._order_totals(items)
            total = check_cents(regular + promoted)
            minimum = check_cents(int(rules.mixed_payment_minimum * 100))
            if total <= minimum:
                raise PolicyUnavailableError(
                    f"Combined payment requires an order above {format_brl(minimum)}; this "
                    f"order totals {format_brl(total)}."
                )
            first = parse_customer_amount(first_amount)
            if not 0 < first < total:
                raise ValueError(
                    f"The first amount must be above zero and below {format_brl(total)}."
                )
            parts = [
                self._split_part(methods[0], first, first_installments),
                self._split_part(methods[1], total - first, second_installments),
            ]
            payments = [payment for payment, _ in parts]
            citations = [*rules.citations, *(c for _, found in parts for c in found)]
            references = _citations(
                tuple(dict.fromkeys((*citations, *self.evidence.citations("mixed_payment"))))
            )
            self.sources.extend(references)
            return {
                "items": lines,
                "order_original_total": format_brl(original_total),
                "total": format_brl(total),
                "payments": payments,
                "sources": references,
                "note": "Combined payments use catalog prices without the PIX discount. These "
                "values must be confirmed when finalizing the purchase. This is not a purchase.",
                **self._metro_shipping(total, total),
            }
        except (ValueError, ArithmeticError) as error:
            return {"error": str(error)}
