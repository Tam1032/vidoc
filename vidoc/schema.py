"""Invoice fields the model extracts, and the arithmetic checks run on them.

Amounts are integers in VND. Dates are ISO (YYYY-MM-DD).
"""

import re
from datetime import date

from pydantic import BaseModel, Field

VAT_RATES = (0, 5, 8, 10)


class LineItem(BaseModel):
    name: str
    quantity: float
    unit_price: int
    amount: int


class Invoice(BaseModel):
    doc_type: str = Field(description="'vat_invoice' (hóa đơn GTGT) or 'retail_receipt' (hóa đơn bán lẻ / biên lai)")
    seller_name: str
    seller_tax_code: str | None = Field(None, description="mã số thuế người bán, 10 or 13 digits")
    invoice_series: str | None = Field(None, description="ký hiệu, e.g. 1C26TAA")
    invoice_number: str | None = Field(None, description="số hóa đơn")
    issue_date: str = Field(description="YYYY-MM-DD")
    buyer_name: str | None = None
    buyer_tax_code: str | None = None
    items: list[LineItem]
    subtotal: int = Field(description="cộng tiền hàng, before VAT")
    vat_rate: int | None = Field(None, description="thuế suất GTGT in percent")
    vat_amount: int = 0
    total: int = Field(description="tổng tiền thanh toán")


def json_schema() -> dict:
    """Schema for guided decoding (vLLM / OpenAI structured outputs)."""
    return Invoice.model_json_schema()


TAX_CODE = re.compile(r"^\d{10}(-\d{3})?$")
FIELDS = ["doc_type", "seller_name", "seller_tax_code", "invoice_series", "invoice_number", "issue_date",
          "buyer_name", "buyer_tax_code", "subtotal", "vat_rate", "vat_amount", "total"]


def check(inv: Invoice, tolerance: int = 1) -> list[str]:
    """Return problems found; an empty list means the invoice is internally consistent.

    Rounding differs between issuers, so sums may be off by `tolerance` VND per line.
    """
    problems = []
    for i, it in enumerate(inv.items):
        if abs(round(it.quantity * it.unit_price) - it.amount) > tolerance:
            problems.append(f"item {i + 1}: {it.quantity} x {it.unit_price} != {it.amount}")
    items_sum = sum(it.amount for it in inv.items)
    if abs(items_sum - inv.subtotal) > tolerance * max(1, len(inv.items)):
        problems.append(f"items add up to {items_sum}, subtotal says {inv.subtotal}")
    if inv.doc_type == "vat_invoice":
        if inv.vat_rate not in VAT_RATES:
            problems.append(f"VAT rate {inv.vat_rate} is not one of {VAT_RATES}")
        elif abs(round(inv.subtotal * inv.vat_rate / 100) - inv.vat_amount) > tolerance:
            problems.append(f"VAT {inv.vat_rate}% of {inv.subtotal} is not {inv.vat_amount}")
        if not inv.seller_tax_code or not TAX_CODE.match(inv.seller_tax_code):
            problems.append(f"seller tax code {inv.seller_tax_code!r} is not 10 or 13 digits")
        if not inv.invoice_number:
            problems.append("invoice number missing")
    if abs(inv.subtotal + inv.vat_amount - inv.total) > tolerance:
        problems.append(f"subtotal {inv.subtotal} + VAT {inv.vat_amount} != total {inv.total}")
    try:
        date.fromisoformat(inv.issue_date)
    except ValueError:
        problems.append(f"issue date {inv.issue_date!r} is not YYYY-MM-DD")
    return problems


def invoice_key(inv: Invoice) -> str:
    """One invoice is identified by seller tax code + series + number (or seller + date + total for receipts)."""
    if inv.seller_tax_code and inv.invoice_number:
        return f"{inv.seller_tax_code}|{inv.invoice_series or ''}|{inv.invoice_number}"
    return f"{inv.seller_name.lower()}|{inv.issue_date}|{inv.total}"
