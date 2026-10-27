"""Hand-coded policy rules that speak the tool-calling protocol.

Lets the agent pipeline (harness, MCP server, eval, CI) run with no model. The
rules encode the sample policy directly, so this is an upper bound for that
policy and a smoke test, not a competitor to the LLM agent: the LLM has to
find and apply the rules from retrieved text, and must keep working when the
policy document changes.
"""

import json
import re

from vidoc.harness.llm import LLMResponse, ToolCall

LIMITS = {"taxi": 500_000, "stationery": 2_000_000, "entertainment": 5_000_000}
CATEGORY_REF = {"entertainment": "2.1", "meals": "2.2", "taxi": "2.3", "hotel": "2.4", "stationery": "2.5",
                "equipment": "2.6"}
COMPANY_TAX_CODE = "0312345678"


def decide(claim: dict, ext: dict, dup: dict) -> tuple[str, str, str]:
    inv, cat, amount = ext["invoice"], claim["category"], claim["amount"]
    note = claim["note"].lower()
    if inv["doc_type"] == "retail_receipt":
        if inv["total"] >= 200_000:
            return "reject", "1.2", "Biên lai không có MST người mua chỉ chấp nhận dưới 200.000 đồng."
    elif inv["total"] >= 200_000 and inv.get("buyer_tax_code") != COMPANY_TAX_CODE:
        return "reject", "1.1", "Hóa đơn không ghi đúng tên và mã số thuế công ty."
    if dup["already_paid"]:
        return "reject", "3.2", "Hóa đơn này đã được thanh toán trước đó."
    if any(re.search(r"rượu|whisky|vodka", it["name"].lower()) for it in inv["items"]):
        return "reject", "2.1", "Chi phí rượu mạnh không được thanh toán."
    if cat == "equipment":
        if inv["total"] >= 10_000_000:
            return "reject", "2.6", "Thiết bị từ 10 triệu phải mua qua quy trình mua sắm tập trung."
        return "escalate", "2.6", "Thiết bị dưới 10 triệu cần trưởng bộ phận xác nhận."
    if any(p.startswith("seller tax code") or "missing" in p for p in ext["problems"]):
        return "escalate", "1.3", "Hóa đơn thiếu thông tin bắt buộc."
    if ext["problems"]:
        return "escalate", "1.4", "Số liệu trên hóa đơn không khớp: " + "; ".join(ext["problems"])
    if (ext["days_from_invoice_to_submission"] or 0) > 30:
        return "escalate", "3.1", "Nộp quá 30 ngày kể từ ngày lập hóa đơn."
    if amount > inv["total"]:
        return "escalate", "3.3", "Số tiền đề nghị lớn hơn tổng tiền hóa đơn."
    if amount > 20_000_000:
        return "escalate", "4.1", "Trên 20 triệu cần kế toán trưởng phê duyệt."
    if cat == "entertainment" and not re.search(r"tiếp khách:|đối tác", note):
        return "escalate", "2.1", "Chi tiếp khách phải ghi tên khách hoặc đối tác."
    if cat == "taxi":
        limit = 800_000 if "sân bay" in note else 500_000
        if inv["total"] > limit:
            return "escalate", "2.3", "Vượt hạn mức taxi."
    elif cat == "hotel":
        nights = sum(it["quantity"] for it in inv["items"] if "phòng" in it["name"].lower()) or 1
        limit = 1_500_000 if re.search(r"hồ chí minh|hà nội", note) else 1_000_000
        if inv["total"] / nights > limit:
            return "escalate", "2.4", "Vượt hạn mức tiền phòng mỗi đêm."
    elif cat == "meals" and inv["total"] > 300_000:
        return "escalate", "2.2", "Vượt phụ cấp ăn uống công tác."
    elif cat in LIMITS and inv["total"] > LIMITS[cat]:
        return "escalate", CATEGORY_REF[cat], "Vượt hạn mức theo loại chi phí."
    return "approve", CATEGORY_REF.get(cat, "4.2"), "Hợp lệ theo quy chế."


class PolicyRules:
    name = "policy-rules"

    async def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        results = {}
        ids = {}
        for m in messages:
            for tc in m.get("tool_calls", []) if m["role"] == "assistant" else []:
                ids[tc["id"]] = tc["function"]["name"]
            if m["role"] == "tool":
                try:
                    results[ids[m["tool_call_id"]]] = json.loads(m["content"])
                except ValueError:
                    results[ids[m["tool_call_id"]]] = {"error": m["content"]}
        n = len(ids)

        def do(name, args):
            return LLMResponse(tool_calls=[ToolCall(f"call_{n}", name, json.dumps(args, ensure_ascii=False))])

        cid = re.search(r"C\d+", messages[1]["content"]).group(0)
        for step in ("get_claim", "extract_invoice", "find_duplicate"):
            if step not in results:
                return do(step, {"claim_id": cid})
        if "search_policy" not in results:
            return do("search_policy", {"query": results["get_claim"]["category"], "k": 3})
        decision, sec, reason = decide(results["get_claim"], results["extract_invoice"], results["find_duplicate"])
        refs = [f"expense_policy#{sec}"]
        tool = {"approve": "approve_expense", "reject": "reject_claim", "escalate": "escalate_claim"}[decision]
        if tool not in results:
            args = {"claim_id": cid, "policy_refs": refs}
            if decision != "approve":
                args["reason"] = reason
            return do(tool, args)
        return do("submit_decision", {"decision": decision, "policy_refs": refs, "summary": reason})
