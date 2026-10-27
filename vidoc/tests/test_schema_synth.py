import random

from vidoc.schema import check, invoice_key
from vidoc.synth import amount_in_words, degrade, make_invoice, render


def test_generated_invoices_are_consistent():
    rng = random.Random(1)
    for _ in range(200):
        inv = make_invoice(rng)
        assert check(inv) == [], inv


def test_check_catches_errors():
    inv = make_invoice(random.Random(2), kind="vat_invoice")
    bad = inv.model_copy(update={"total": inv.total + 5000, "seller_tax_code": "12AB"})
    problems = check(bad)
    assert any("total" in p for p in problems) and any("tax code" in p for p in problems)


def test_render_and_degrade():
    rng = random.Random(3)
    for kind in ("vat_invoice", "retail_receipt"):
        img = degrade(render(make_invoice(rng, kind=kind), rng), rng)
        assert img.size[0] > 400 and img.size[1] > 200


def test_amount_in_words():
    assert amount_in_words(12_500_000) == "Mười hai triệu năm trăm nghìn đồng"
    assert amount_in_words(215_000) == "Hai trăm mười lăm nghìn đồng"
    assert amount_in_words(21_000_000) == "Hai mươi mốt triệu đồng"


def test_invoice_key_for_receipts_and_vat():
    rng = random.Random(4)
    vat, rec = make_invoice(rng, kind="vat_invoice"), make_invoice(rng, kind="retail_receipt")
    assert invoice_key(vat).startswith(vat.seller_tax_code)
    assert str(rec.total) in invoice_key(rec)
