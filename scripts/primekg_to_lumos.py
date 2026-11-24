#!/usr/bin/env python3
"""Convert PrimeKG into the CSV edge list that Lumos's getGraph reads.

    python scripts/primekg_to_lumos.py data/primekg            # writes data/primekg/primekg.csv

Input (from https://doi.org/10.7910/DVN/IXA7BM): kg.csv, plus the optional
drug_features.csv and disease_features.csv for node context.

Output, one row per undirected node pair:
    src_id, src_type, src_name, src_context, dst_id, dst_type, dst_name, dst_context, relation
so every node has L = [type, name, context] (context is the long description,
kept last as the Lumos convention expects) and every edge L = [relation].

kg.csv lists each edge in both directions; pairs are written once. A pair
with several relations keeps the first one in PRIORITY (others: the first
seen). Relations are renamed with --rename, by default to the plain words
used in prime.lumos: indication -> treat, contraindication -> contraindicate.
"""

import argparse
import csv
import sys
from pathlib import Path

PRIORITY = ["indication", "contraindication", "off-label use"]
DEFAULT_RENAMES = {"indication": "treat", "contraindication": "contraindicate"}
DRUG_CONTEXT = ["description", "indication", "mechanism_of_action"]
DISEASE_CONTEXT = ["mondo_definition", "umls_description", "orphanet_definition"]


def read_contexts(path: Path, columns):
    """node_index -> the first non-empty of `columns`."""
    if not path.exists():
        return {}
    csv.field_size_limit(sys.maxsize)
    contexts = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            text = next((row[c].strip() for c in columns if row.get(c, "").strip()), "")
            if text:
                contexts[row["node_index"]] = " ".join(text.split())
    return contexts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", help="directory with kg.csv (and the feature files)")
    parser.add_argument("--output", default=None, help="output CSV (default: <directory>/primekg.csv)")
    parser.add_argument("--rename", action="append", default=None, metavar="OLD=NEW",
                        help="rename a relation; repeatable (default: indication=treat, contraindication=contraindicate)")
    args = parser.parse_args()

    directory = Path(args.directory)
    output = Path(args.output) if args.output else directory / "primekg.csv"
    renames = DEFAULT_RENAMES if args.rename is None else dict(r.split("=", 1) for r in args.rename)
    contexts = {**read_contexts(directory / "drug_features.csv", DRUG_CONTEXT),
                **read_contexts(directory / "disease_features.csv", DISEASE_CONTEXT)}
    rank = {relation: i for i, relation in enumerate(PRIORITY)}

    nodes, edges, merged = {}, {}, 0
    with open(directory / "kg.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for side in ("x", "y"):
                index = row[f"{side}_index"]
                if index not in nodes:
                    nodes[index] = (row[f"{side}_type"], row[f"{side}_name"], contexts.get(index, ""))
            a, b = row["x_index"], row["y_index"]
            key = (a, b) if a <= b else (b, a)
            relation = row["relation"]
            previous = edges.get(key)
            if previous is None:
                edges[key] = relation
            elif previous != relation:
                merged += 1
                if rank.get(relation, len(rank)) < rank.get(previous, len(rank)):
                    edges[key] = relation

    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["src_id", "src_type", "src_name", "src_context",
                         "dst_id", "dst_type", "dst_name", "dst_context", "relation"])
        for (a, b), relation in edges.items():
            writer.writerow([a, *nodes[a], b, *nodes[b], renames.get(relation, relation)])

    with_context = sum(1 for n in nodes.values() if n[2])
    print(f"wrote {output}: {len(nodes)} nodes ({with_context} with context), {len(edges)} edges; "
          f"{merged} rows had a second relation for an existing pair")


if __name__ == "__main__":
    main()
