"""
Find people in the database who are probably the same person and merge them.

python merge_duplicates.py people.db               # dry run: show what would be merged
python merge_duplicates.py people.db --apply       # back up the file, then merge
python merge_duplicates.py people.db --thresh 0.45 --apply

Needs database.py in the same folder.
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from database import FaceDB


def similarity(db, a, b):
    return float(db.people[a]["vec"] @ db.people[b]["vec"])


def plan_merges(db, thresh):
    # Named and well-sampled people are kept; the rest are folded into them.
    order = sorted(db.people, key=lambda pid: (db.people[pid]["name"] is None,
                                               -db.people[pid]["n"], pid))
    absorbed, plan, blocked = set(), [], []
    for i, keep in enumerate(order):
        if keep in absorbed:
            continue
        for other in order[i + 1:]:
            if other in absorbed:
                continue
            sim = similarity(db, keep, other)
            if sim < thresh:
                continue
            a, b = db.people[keep]["name"], db.people[other]["name"]
            if a and b and a != b:
                blocked.append((keep, other, sim))
                continue
            plan.append((keep, other, sim))
            absorbed.add(other)
    return plan, blocked


def near_misses(db, thresh, margin=0.1):
    ids = sorted(db.people)
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            sim = similarity(db, a, b)
            if thresh - margin <= sim < thresh:
                pairs.append((a, b, sim))
    return sorted(pairs, key=lambda t: -t[2])


def main():
    ap = argparse.ArgumentParser(description="Merge duplicate people in a face database.")
    ap.add_argument("db", help="Path to the SQLite database")
    ap.add_argument("--thresh", type=float, default=0.5,
                    help="Similarity needed to merge two people (default: 0.5, stricter "
                         "than the 0.4 used for recognition)")
    ap.add_argument("--apply", action="store_true",
                    help="Actually merge (a backup copy is made first). Without it, dry run")
    args = ap.parse_args()

    if not Path(args.db).exists():
        sys.exit(f"Database not found: {args.db}")

    db = FaceDB(args.db)
    plan, blocked = plan_merges(db, args.thresh)
    print(f"{len(db.people)} people in {args.db}, merge threshold {args.thresh}\n")

    if plan:
        print("Will merge:")
        for keep, other, sim in plan:
            print(f"  {db.label(other)} -> {db.label(keep)}  (similarity {sim:.2f})")
    else:
        print("Nothing to merge.")

    if blocked:
        print("\nSkipped (both have different names):")
        for keep, other, sim in blocked:
            print(f"  {db.label(keep)} / {db.label(other)}  (similarity {sim:.2f})")

    close = near_misses(db, args.thresh)
    if close:
        print(f"\nClose but below {args.thresh} (not merged):")
        for a, b, sim in close:
            print(f"  {db.label(a)} / {db.label(b)}  (similarity {sim:.2f})")

    if not plan:
        db.close()
        return
    if not args.apply:
        print("\nDry run only. Add --apply to merge (the file is backed up first).")
        db.close()
        return

    backup = f"{args.db}.bak-{datetime.now():%Y%m%d-%H%M%S}"
    shutil.copy2(args.db, backup)
    print(f"\nBackup saved to {backup}")
    for keep, other, _ in plan:
        db.merge(keep, other)
    db.close()
    print(f"Merged {len(plan)} duplicate(s). Run --list-db to see the result.")


if __name__ == "__main__":
    main()