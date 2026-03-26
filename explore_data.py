#!/usr/bin/env python3

import json
import ast

from collections import Counter


def load_companies(path="companies.jsonl"):
    companies = []
    with open(path) as f:
        for line in f:
            companies.append(json.loads(line))
    return companies


def parse_naics(val):
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return None
    return None


def section(title):
    print(f"\n--- {title} ---")


def main():
    companies = load_companies()

    section(f"TOTAL COMPANY COUNT: {len(companies)}")

    # --- Field completeness ---
    section("FIELD COMPLETENESS")
    fields = [
        "operational_name", "website", "description", "address",
        "revenue", "employee_count", "year_founded", "is_public",
        "business_model", "primary_naics", "secondary_naics",
        "core_offerings", "target_markets",
    ]
    for field in fields:
        non_null = sum(1 for c in companies if c.get(
            field) not in [None, [], ""])
        pct = non_null / len(companies) * 100
        flag = " !" if pct < 70 else ""
        print(f"  {field:<20} {non_null:>4}/{len(companies)}  ({pct:5.1f}%){flag}")

    # --- Country distribution ---
    section("COUNTRY DISTRIBUTION (top 25)")
    country_counter = Counter()
    for c in companies:
        addr = c.get("address")
        cc = addr.get("country_code") if isinstance(addr, dict) else None
        country_counter[cc] += 1
    for k, v in country_counter.most_common(25):
        print(f"  {str(k):<6} {v:>4}")

    # --- NAICS sector distribution ---
    section("NAICS SECTOR DISTRIBUTION")
    naics_counter = Counter()
    sector_labels = {}
    for c in companies:
        n = parse_naics(c.get("primary_naics"))
        if n:
            code = n.get("code", "")
            sector = code[:2]
            naics_counter[sector] += 1
            if sector not in sector_labels:
                sector_labels[sector] = n.get("label", "")[:45]
    for k, v in naics_counter.most_common():
        flag = " !" if v < 5 else ""
        print(f"  Sector {k:<4} ({sector_labels.get(k, ''):<45}) {v:>4}{flag}")

    # --- is_public ---
    section("IS_PUBLIC RATIO")
    pub = Counter(c.get("is_public") for c in companies)
    for k, v in pub.items():
        print(f"  {str(k):<6} {v:>4}  ({v / len(companies) * 100:.1f}%)")

    # --- Business model vocabulary ---
    section("BUSINESS MODEL VOCABULARY")
    bm_counter = Counter()
    for c in companies:
        for bm in (c.get("business_model") or []):
            bm_counter[bm] += 1
    for k, v in bm_counter.most_common():
        print(f"  {v:>4}  {k}")

    # --- Revenue distribution ---
    section("REVENUE DISTRIBUTION")
    revs = sorted(c["revenue"]
                  for c in companies if c.get("revenue") is not None)
    n = len(revs)
    if n:
        print(f"  count:   {n} ({n / len(companies) * 100:.1f}% non-null)")
        print(f"  min:     ${min(revs):>20,.0f}")
        print(f"  p25:     ${revs[n // 4]:>20,.0f}")
        print(f"  median:  ${revs[n // 2]:>20,.0f}")
        print(f"  p75:     ${revs[3 * n // 4]:>20,.0f}")
        print(f"  max:     ${max(revs):>20,.0f}")
        print(f"  > $50M:  {sum(1 for r in revs if r > 50_000_000):>4}")
        print(f"  > $1B:   {sum(1 for r in revs if r > 1_000_000_000):>4}")
        print(
            f"  > $10T:  {sum(1 for r in revs if r > 10_000_000_000_000):>4}")

    # --- Employee count distribution ---
    section("EMPLOYEE COUNT DISTRIBUTION")
    emps = sorted(c["employee_count"]
                  for c in companies if c.get("employee_count") is not None)
    n = len(emps)
    if n:
        print(f"  count:   {n} ({n / len(companies) * 100:.1f}% non-null)")
        print(f"  min:     {min(emps):>10.0f}")
        print(f"  p25:     {emps[n // 4]:>10.0f}")
        print(f"  median:  {emps[n // 2]:>10.0f}")
        print(f"  p75:     {emps[3 * n // 4]:>10.0f}")
        print(f"  max:     {max(emps):>10.0f}")
        print(f"  > 1000:  {sum(1 for e in emps if e > 1000):>4}")

    # --- Zero searchable text ---
    section("COMPANIES WITH ZERO SEARCHABLE TEXT")
    zero = [
        c for c in companies
        if not c.get("description") and not c.get("core_offerings") and not parse_naics(c.get("primary_naics"))
    ]
    print(f"  {len(zero)}")
    for c in zero:
        print(f"    - {c.get('operational_name')}")

    # --- primary_naics type check ---
    section("PRIMARY_NAICS STORAGE FORMAT")
    types = Counter(type(c.get("primary_naics")).__name__ for c in companies)
    for k, v in types.items():
        flag = " !" if k == "str" else ""
        print(f"  {k}: {v}{flag}")

    print()


if __name__ == "__main__":
    main()
