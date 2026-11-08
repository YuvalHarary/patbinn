#!/usr/bin/env python3
"""
Regenerate docs/TAXONOMY.md: the 58 evaluated classes, what "variant" means in
each family, and the NCBI accessions behind every class.

The per-isolate detail (lengths, deflines, fold assignment) lives in
docs/GENOME_INVENTORY.txt. This file is the compact class-level view, which is
what you want when checking which organisms a reported number covers.

    python3 scripts/make_taxonomy.py
"""

import collections
import csv
import os

MANIFEST = "data/manifests/genome_manifest_multi.csv"
OUT = "docs/TAXONOMY.md"

# The family name recorded at retrieval time, and the name ICTV uses now.
# The manifests keep the retrieval-time name so the stored paths stay valid.
CURRENT_NAME = {"Herpesviridae": "Orthoherpesviridae"}

# What one "variant" is in each family. These ranks genuinely differ: the label
# means a species in some families, a genotype or serotype in others. Stated
# per family so a reported per-class number is not read as a uniform rank.
RANK = {
    "Coronaviridae": (
        "Distinct human coronaviruses. Six are separate ICTV species; "
        "SARS-CoV-1 and SARS-CoV-2 are two viruses within the single species "
        "*Severe acute respiratory syndrome-related coronavirus*."),
    "Flaviviridae": (
        "Separate species. The four dengue classes are the four dengue "
        "serotypes, which ICTV also treats as four species."),
    "Filoviridae": (
        "Mostly separate species: the five ebolaviruses and Lloviu cuevavirus. "
        "Marburg and Ravn are two viruses within the single species "
        "*Marburg marburgvirus*."),
    "Hepadnaviridae": (
        "Genotypes A to H of one species, hepatitis B virus. This is the "
        "finest-grained split in the taxonomy, and the only family whose "
        "classes are all one species."),
    "Retroviridae": (
        "Two HIV-1 subtypes (clades B and C of one species) plus HIV-2, "
        "a separate species."),
    "Herpesviridae": (
        "Separate species, including HHV-6A and HHV-6B, which ICTV separated "
        "into two species in 2012."),
    "Picornaviridae": (
        "Mixed. Poliovirus 1 to 3 are three serotypes of *Enterovirus C*; "
        "Coxsackievirus A16 and B3 belong to *Enterovirus A* and "
        "*Enterovirus B*; rhinovirus A, B and C are three species."),
    "Paramyxoviridae": (
        "Separate species, except RSV-A and RSV-B, which are two subgroups of "
        "human respiratory syncytial virus."),
}

# A branch whose classes do not all belong to the family it is named for.
# Flagged rather than renamed: the grouping is what the router was trained on.
FAMILY_NOTE = {
    "Paramyxoviridae": (
        "This branch spans two families as currently classified. RSV-A, RSV-B "
        "and human metapneumovirus were moved to *Pneumoviridae* by ICTV in "
        "2016; the remaining five classes are *Paramyxoviridae*. The branch is "
        "a grouping of the classifier, named for the family most of its "
        "classes belong to, and is called a family throughout the code and the "
        "manifests."),
}


def accession_of(row):
    """The accession is the first token of the stored defline.

    The manifest's own `accession` column holds the Entrez UID, not the
    accession.version string, so take the defline token instead.
    """
    defline = (row.get("defline") or "").strip()
    return defline.split()[0] if defline else f"uid:{row['accession']}"


def main():
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"missing {MANIFEST}; run from the repository root")

    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))

    by_family = collections.OrderedDict()
    for r in rows:
        by_family.setdefault(r["family"], collections.OrderedDict()) \
                 .setdefault(r["variant"], []).append(r)

    n_classes = sum(len(v) for v in by_family.values())
    out = []
    out.append("# Evaluated taxonomy\n")
    out.append(
        f"The {n_classes} classes the classifier resolves, across "
        f"{len(by_family)} families and {len(rows)} NCBI isolates, with the "
        "accessions behind each class.\n")
    out.append(
        "Per-isolate detail (sequence length, defline, fold) is in "
        "[GENOME_INVENTORY.txt](GENOME_INVENTORY.txt). Regenerate this file "
        "with `python3 scripts/make_taxonomy.py`.\n")

    out.append('\n## What "variant" means\n')
    out.append(
        "The label is not a single taxonomic rank. It is the level at which a "
        "family is clinically distinguished, which differs between families, "
        "so a per-class number should be read against the rank below rather "
        "than as a uniform one.\n")
    out.append("| Family | Classes | Isolates | What one class is |")
    out.append("|---|---|---|---|")
    for fam, variants in by_family.items():
        n_iso = sum(len(v) for v in variants.values())
        mark = " [^1]" if fam in FAMILY_NOTE else ""
        out.append(f"| {CURRENT_NAME.get(fam, fam)}{mark} | {len(variants)} | "
                   f"{n_iso} | {RANK.get(fam, '')} |")

    for fam, note in FAMILY_NOTE.items():
        out.append(f"\n[^1]: **{CURRENT_NAME.get(fam, fam)}.** {note}\n")

    out.append("\n## Classes and accessions\n")
    for fam, variants in by_family.items():
        shown = CURRENT_NAME.get(fam, fam)
        note = f" (retrieved as *{fam}*)" if shown != fam else ""
        n_iso = sum(len(v) for v in variants.values())
        out.append(f"\n### {shown}{note}\n")
        out.append(f"{len(variants)} classes, {n_iso} isolates.\n")
        out.append("| Class | Isolates | Accessions |")
        out.append("|---|---|---|")
        for var, rs in variants.items():
            accs = ", ".join(f"`{accession_of(r)}`" for r in rs)
            out.append(f"| {var} | {len(rs)} | {accs} |")

    out.append(
        "\n## Naming\n\n"
        "Family names are those used when the genomes were retrieved, so that "
        "the stored paths and manifests stay valid. Where ICTV now uses a "
        "different name, the current one is given above and the retrieval-time "
        "name is noted beside it.\n")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {OUT}: {n_classes} classes, {len(rows)} isolates, "
          f"{len(by_family)} families")


if __name__ == "__main__":
    main()
