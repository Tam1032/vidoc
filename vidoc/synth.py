"""Synthetic Vietnamese invoices and receipts with exact labels.

Real scanned invoices with field labels are scarce, so training and the first
evals use generated ones: VAT invoices (hóa đơn GTGT) and retail receipts,
rendered with varied layouts and degraded like phone photos (rotation, blur,
noise, JPEG). Mix in real data (e.g. MC-OCR 2021 receipts) before trusting
any accuracy number.

    python -m vidoc.synth --n 3000 --out data/synth
"""

import argparse
import io
import json
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from vidoc.schema import Invoice, LineItem

FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
             "/System/Library/Fonts/Supplemental", "C:/Windows/Fonts"]
FONT_NAMES = ["DejaVuSans.ttf", "DejaVuSerif.ttf", "LiberationSans-Regular.ttf", "LiberationSerif-Regular.ttf",
              "Arial.ttf", "Times New Roman.ttf"]

SELLERS = [
    ("CÔNG TY TNHH NHÀ HÀNG BẾP VIỆT", "meals"), ("CÔNG TY CP DỊCH VỤ ẨM THỰC SÀI GÒN XANH", "meals"),
    ("CÔNG TY TNHH TAXI ÁNH DƯƠNG", "taxi"), ("CÔNG TY CP DI CHUYỂN NHANH", "taxi"),
    ("CÔNG TY TNHH VĂN PHÒNG PHẨM HỒNG HÀ PHÁT", "stationery"), ("CÔNG TY TNHH THIẾT BỊ SỐ MINH KHANG", "equipment"),
    ("CÔNG TY CP KHÁCH SẠN BIỂN NHỚ", "hotel"), ("CÔNG TY TNHH CÀ PHÊ PHỐ CŨ", "meals"),
]
SHOPS = [("QUÁN CƠM TẤM BA GHIỀN", "meals"), ("TẠP HÓA CÔ BA", "stationery"), ("PHỞ HÀ NỘI 24", "meals"),
         ("NHÀ SÁCH NHÂN VĂN", "stationery")]
ITEMS = {
    "meals": [("Lẩu thái hải sản", 350_000, 650_000), ("Cơm chiên dương châu", 60_000, 120_000),
              ("Bia Sài Gòn", 18_000, 35_000), ("Nước suối", 10_000, 20_000), ("Gỏi cuốn tôm thịt", 45_000, 90_000),
              ("Cà phê sữa đá", 25_000, 55_000), ("Bò lúc lắc", 150_000, 280_000)],
    "taxi": [("Cước vận chuyển hành khách", 80_000, 450_000), ("Phí cầu đường", 15_000, 40_000)],
    "stationery": [("Giấy A4 Double A", 75_000, 95_000), ("Bút bi Thiên Long", 4_000, 8_000),
                   ("Bìa hồ sơ", 6_000, 15_000), ("Mực in HP 680", 250_000, 420_000)],
    "equipment": [("Chuột không dây Logitech", 250_000, 650_000), ("Bàn phím cơ", 900_000, 2_500_000),
                  ("Màn hình 24 inch", 2_800_000, 4_900_000), ("Laptop văn phòng", 14_000_000, 22_000_000)],
    "hotel": [("Tiền phòng Deluxe", 900_000, 1_800_000), ("Ăn sáng", 120_000, 200_000)],
}
STREETS = ["Lê Lợi", "Nguyễn Huệ", "Trần Hưng Đạo", "Hai Bà Trưng", "Lý Thường Kiệt", "Võ Văn Tần", "Cách Mạng Tháng 8"]
CITIES = ["Quận 1, TP. Hồ Chí Minh", "Quận 3, TP. Hồ Chí Minh", "Quận Hoàn Kiếm, Hà Nội", "Quận Hải Châu, Đà Nẵng"]
BUYER = ("CÔNG TY TNHH GIẢI PHÁP SỐ ABC", "0312345678")
OTHER_BUYERS = [("CÔNG TY CP XYZ VIỆT NAM", "0109876543"), ("Khách lẻ", None)]

DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def _three(n: int, full: bool) -> str:
    h, t, u = n // 100, n // 10 % 10, n % 10
    out = []
    if full or h:
        out.append(f"{DIGITS[h]} trăm")
    if t == 0 and u and (full or h):
        out.append("lẻ")
    if t == 1:
        out.append("mười")
    elif t > 1:
        out.append(f"{DIGITS[t]} mươi")
    if u:
        if u == 1 and t > 1:
            out.append("mốt")
        elif u == 5 and t > 0:
            out.append("lăm")
        else:
            out.append(DIGITS[u])
    return " ".join(out)


def amount_in_words(n: int) -> str:
    """12_500_000 -> 'Mười hai triệu năm trăm nghìn đồng'."""
    if n == 0:
        return "Không đồng"
    units = ["", "nghìn", "triệu", "tỷ"]
    groups = []
    while n:
        groups.append(n % 1000)
        n //= 1000
    words = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g:
            words.append((_three(g, full=i < len(groups) - 1) + " " + units[i]).strip())
    text = " ".join(words)
    return text[0].upper() + text[1:] + " đồng"


def money(n: int, sep: str = ".") -> str:
    return f"{n:,}".replace(",", sep)


def _tax_code(rng: random.Random) -> str:
    code = str(rng.randint(10**9, 10**10 - 1))
    return code if rng.random() > 0.15 else f"{code}-{rng.randint(1, 9):03d}"


def make_invoice(rng: random.Random, kind: str | None = None, category: str | None = None,
                 start: date = date(2026, 1, 1)) -> Invoice:
    kind = kind or rng.choices(["vat_invoice", "retail_receipt"], [0.7, 0.3])[0]
    pool = SELLERS if kind == "vat_invoice" else SHOPS
    pool = [s for s in pool if category is None or s[1] == category] or pool
    seller, cat = rng.choice(pool)
    lines = []
    for name, lo, hi in rng.sample(ITEMS[cat], k=rng.randint(1, min(4, len(ITEMS[cat])))):
        price = round(rng.randint(lo, hi), -3)
        qty = rng.choice([1, 1, 1, 2, 3, 5, 10]) if cat in ("meals", "stationery") and price < 100_000 else 1
        lines.append(LineItem(name=name, quantity=qty, unit_price=price, amount=qty * price))
    subtotal = sum(it.amount for it in lines)
    issue = start + timedelta(days=rng.randint(0, 300))
    if kind == "vat_invoice":
        rate = rng.choice([8, 10, 10, 5]) if cat != "equipment" else 10
        vat = round(subtotal * rate / 100)
        buyer = BUYER if rng.random() < 0.85 else rng.choice(OTHER_BUYERS)
        return Invoice(doc_type=kind, seller_name=seller, seller_tax_code=_tax_code(rng),
                       invoice_series=f"1C{issue.year % 100}T{rng.choice('ABCDE')}{rng.choice('ABCDE')}",
                       invoice_number=f"{rng.randint(1, 99999):08d}", issue_date=issue.isoformat(),
                       buyer_name=buyer[0], buyer_tax_code=buyer[1], items=lines, subtotal=subtotal,
                       vat_rate=rate, vat_amount=vat, total=subtotal + vat)
    return Invoice(doc_type=kind, seller_name=seller, issue_date=issue.isoformat(), items=lines,
                   subtotal=subtotal, vat_amount=0, total=subtotal)


def _font(rng: random.Random, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = (["DejaVuSans-Bold.ttf", "DejaVuSerif-Bold.ttf", "LiberationSans-Bold.ttf"] if bold else []) + FONT_NAMES
    rng_names = names[:]
    rng.shuffle(rng_names)
    for name in (names[:1] if bold else []) + rng_names:
        for d in FONT_DIRS:
            p = Path(d) / name
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size)


def render(inv: Invoice, rng: random.Random) -> Image.Image:
    vat = inv.doc_type == "vat_invoice"
    width = rng.choice([1100, 1240]) if vat else rng.choice([560, 620])
    img = Image.new("RGB", (width, 2400), "white")
    d = ImageDraw.Draw(img)
    size = rng.randint(20, 24) if vat else rng.randint(18, 22)
    f, fb, fh = _font(rng, size), _font(rng, size, True), _font(rng, size + 10, True)
    sep = rng.choice([".", ","])
    y, x = 40, 40

    def line(text, font=f, dx=0, center=False):
        nonlocal y
        tx = (width - d.textlength(text, font=font)) / 2 if center else x + dx
        d.text((tx, y), text, fill=(20, 20, 20), font=font)
        y += int(font.size * 1.5)

    dt = date.fromisoformat(inv.issue_date)
    if vat:
        line(inv.seller_name, fb)
        line(f"Mã số thuế: {inv.seller_tax_code}")
        line(f"Địa chỉ: {rng.randint(1, 300)} {rng.choice(STREETS)}, {rng.choice(CITIES)}")
        y += 10
        line("HÓA ĐƠN GIÁ TRỊ GIA TĂNG", fh, center=True)
        line(f"Ngày {dt.day:02d} tháng {dt.month:02d} năm {dt.year}", center=True)
        line(f"Ký hiệu: {inv.invoice_series}    Số: {inv.invoice_number}", center=True)
        y += 10
        line(f"Tên đơn vị mua hàng: {inv.buyer_name or ''}")
        line(f"Mã số thuế: {inv.buyer_tax_code or ''}")
        line(f"Hình thức thanh toán: {rng.choice(['TM/CK', 'Chuyển khoản', 'Tiền mặt'])}")
    else:
        line(inv.seller_name, fb, center=True)
        line(f"{rng.randint(1, 300)} {rng.choice(STREETS)}", center=True)
        line("HÓA ĐƠN BÁN LẺ", fh, center=True)
        line(f"Ngày: {dt.day:02d}/{dt.month:02d}/{dt.year}  {rng.randint(7, 21):02d}:{rng.randint(0, 59):02d}", center=True)
    y += 10
    d.line((x, y, width - x, y), fill="black", width=2)
    y += 12
    cols = [x, int(width * 0.52), int(width * 0.62), int(width * 0.80)] if vat else [x, int(width * 0.55), int(width * 0.64), int(width * 0.80)]
    for c, h in zip(cols, ["Tên hàng hóa, dịch vụ", "SL", "Đơn giá", "Thành tiền"] if vat else ["Món", "SL", "Giá", "T.Tiền"]):
        d.text((c, y), h, fill="black", font=fb)
    y += int(size * 1.6)
    for it in inv.items:
        q = f"{it.quantity:g}"
        # Long item names wrap onto extra lines inside the name column, as on real invoices.
        name_lines, cur = [], ""
        for word in it.name.split():
            if cur and d.textlength(f"{cur} {word}", font=f) > cols[1] - cols[0] - 10:
                name_lines.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
        name_lines.append(cur)
        for c, t in zip(cols, [name_lines[0], q, money(it.unit_price, sep), money(it.amount, sep)]):
            d.text((c, y), t, fill=(20, 20, 20), font=f)
        for extra in name_lines[1:]:
            y += int(size * 1.3)
            d.text((cols[0], y), extra, fill=(20, 20, 20), font=f)
        y += int(size * 1.5)
    d.line((x, y, width - x, y), fill="black", width=1)
    y += 12
    right = cols[2]
    if vat:
        for label, val in [("Cộng tiền hàng:", inv.subtotal), (f"Thuế suất GTGT: {inv.vat_rate}%   Tiền thuế GTGT:", inv.vat_amount),
                           ("Tổng cộng tiền thanh toán:", inv.total)]:
            d.text((x, y), label, fill="black", font=fb if "Tổng" in label else f)
            d.text((cols[3], y), money(val, sep), fill="black", font=fb)
            y += int(size * 1.6)
        line(f"Số tiền viết bằng chữ: {amount_in_words(inv.total)}.")
    else:
        d.text((x, y), "TỔNG CỘNG:", fill="black", font=fb)
        d.text((right, y), money(inv.total, sep) + " đ", fill="black", font=fb)
        y += int(size * 1.6)
        line("Cảm ơn quý khách!", center=True)
    return img.crop((0, 0, width, min(2400, y + 60)))


def degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    """Make a clean render look like a phone photo."""
    if rng.random() < 0.7:
        img = img.rotate(rng.uniform(-3, 3), expand=True, fillcolor=(rng.randint(180, 240),) * 3)
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    arr = np.asarray(img).astype(np.float32)
    arr = arr * rng.uniform(0.8, 1.05) + rng.uniform(-15, 10)
    if rng.random() < 0.6:
        arr += np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(2, 9), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(45, 90))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def generate(n: int, out: Path, seed: int = 0, splits=(0.8, 0.1, 0.1)) -> None:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    files = {s: (out / f"{s}.jsonl").open("w") for s in ("train", "val", "test")}
    for i in range(n):
        inv = make_invoice(rng)
        img = degrade(render(inv, rng), rng)
        name = f"inv_{i:05d}.jpg"
        img.save(out / "images" / name, quality=92)
        r = rng.random()
        split = "train" if r < splits[0] else "val" if r < splits[0] + splits[1] else "test"
        files[split].write(json.dumps({"image": f"images/{name}", "label": inv.model_dump()}, ensure_ascii=False) + "\n")
    for f in files.values():
        f.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--out", default="data/synth")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    generate(a.n, Path(a.out), a.seed)


if __name__ == "__main__":
    main()
