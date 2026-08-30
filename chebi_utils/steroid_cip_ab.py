"""Demo: C5 alpha/beta from get_steroid_positions.

Run: python -m chebi_utils.steroid_cip_ab
"""

from chebi_utils.extract_properties import get_steroid_positions
from chebi_utils.read_molecule import smiles_or_inchi_to_mol


def _c5_ab_label(positions: dict[str, list]) -> str:
    if "steroid_5_alpha" in positions:
        return "alpha"
    if "steroid_5_beta" in positions:
        return "beta"
    return "-"


# Paste substances here: (name, SMILES_or_InChI, "alpha"|"beta")
EXAMPLES: list[tuple[str, str, str]] = [
    ("3-oxo-5β-steroid", "*C1CCC2C3CC[C@]4([H])CC(=O)CCC4(C)C3CCC12C", "beta"),
    ("5beta-androstane-3beta,17alpha-diol", "[H][C@]12CC[C@]3([H])[C@]([H])(CC[C@]4(C)[C@H](O)CC[C@@]34[H])[C@@]1(C)CC[C@H](O)C2", "beta"),
    ("5beta-tetrahydro-11-deoxycorticosterone", "[H][C@]12CC[C@]3([H])[C@]([H])(CC[C@@]4(C)[C@@]3([H])CC[C@]4([H])C(=O)CO)[C@@]1(C)CC[C@@H](O)C2", "beta"),
    ("5β-chola-3,8(14),11-trien-24-oic acid", "[H][C@@]12C=CCC[C@]1(C)[C@@]1([H])C=C[C@@]3(C)C(=C1CC2)CC[C@]3([H])[C@H](C)CCC(=O)O", "beta"),
    ("5β-chola-3,8(14),11-trien-24-oic acid", "[H][C@@]12C=CCC[C@]1(C)[C@@]1([H])C=C[C@@]3(C)C(=C1CC2)CC[C@]3([H])[C@H](C)CCC(=O)O", "beta"),
    ("andrastin A", "[H][C@@]12C=C(C)[C@@]3(C)C(=O)C(C)=C(O)[C@@]3(C(=O)OC)[C@@]1(C)CC[C@]1([H])C(C)(C)[C@@H](OC(C)=O)CC[C@@]21C=O", "beta"),
    ("andrastin B", "[H][C@@]12C=C(C)[C@@]3(C)C(=O)C(C)=C(O)[C@@]3(C(=O)OC)[C@@]1(C)CC[C@]1([H])C(C)(C)[C@@H](OC(C)=O)CC[C@@]21CO", "beta"),
    ("andrastin C", "[H][C@@]12C=C(C)[C@@]3(C)C(=O)C(C)=C(O)[C@@]3(C(=O)OC)[C@@]1(C)CC[C@]1([H])C(C)(C)[C@@H](OC(C)=O)CC[C@@]21C", "beta"),
    ("andrastin D", "[H][C@@]12C=C(C)[C@@]3(C)C(=O)C(C)=C(O)[C@@]3(C(=O)OC)[C@@]1(C)CC[C@]1([H])C(C)(C)C(=O)CC[C@@]21C", "beta"),
    ("andrastin E", "[H][C@@]12C=C(C)[C@@]3(C)C(=O)C(C)=C(O)[C@@]3(C(=O)OC)[C@@]1(C)CC[C@]1([H])C(C)(C)[C@H](O)CC[C@@]21C", "beta"),
    ("andrastin F", "[H][C@@]12C=C(C)[C@@]3(C)C(=O)C(C)=C(O)[C@@]3(C(=O)OC)[C@@]1(C)CC[C@]1([H])C(C)(C)[C@@H](O)CC[C@@]21C", "beta"),
    ("Tetrahydro-11-dehydrocorticosterone", "[H][C@]12CC[C@@]3([H])[C@]4([H])CC[C@]([H])(C(=O)CO)[C@@]4(C)CC(=O)[C@]3([H])[C@@]1(C)CC[C@@H](O)C2", "beta"),
    ("25S)-5alpha-cholestan..", "[H][C@@]12[C@@H](O)[C@@H](O)[C@]3(O)[C@]([H])(CC[C@]4(C)[C@@]([H])([C@H](C)CCC[C@H](C)CO)[C@@H](O)[C@@H](O)[C@]43[H])[C@@]1(C)CC[C@H](O)[C@@H]2O", "alpha"),
    ("(25S)-5alpha-cholestan-3beta ...", "[H][C@@]12[C@@H](O)C[C@]3(O)[C@]([H])(CC[C@]4(C)[C@@]([H])([C@H](C)CCC[C@H](C)CO)[C@@H](O)[C@H](O)[C@]43[H])[C@@]1(C)CC[C@H](O)[C@@H]2O", "alpha"),
    ("(25S)-5alpha-cholestan-3beta,6alpha,7beta,8", "[H][C@]1([C@H](C)CCC[C@H](C)CO)[C@@H](O)[C@H](O)[C@]2([H])[C@]1(C)CC[C@]1([H])[C@@]3(C)CC[C@H](O)C[C@]3([H])[C@@H](O)[C@H](O)[C@]12O", "alpha"),
    ("(25S)-5alpha-cholestan-3beta,6al...", "[H][C@]1([C@H](C)CCC[C@H](C)CO)[C@@H](O)[C@@H](O)[C@]2([H])[C@]1(C)CC[C@]1([H])[C@@]3(C)CC[C@H](O)C[C@]3([H])[C@@H](O)C[C@]12O", "alpha"),
    ("(25S)-5alpha-cholestan-3beta,6bet..", "[H][C@]1([C@H](C)CCC[C@H](C)CO)[C@@H](O)[C@H](O)[C@@]2([H])[C@]3([H])C[C@@H](O)[C@@]4([H])C[C@@H](O)CC[C@]4(C)[C@@]3([H])CC[C@]12C", "alpha"),
    ("5alpha-Dermophol", "[H][C@]12C[C@H](O)CC[C@]1(C)[C@@]1([H])C[C@H](O)[C@@]3(C)[C@@]([H])(CC[C@]3([H])[C@H](C)CCCC(O)(CO)CO)[C@]1([H])[C@H](O)C2", "alpha"),
    ("5alpha-Pregn-2-en-20-one", "[H][C@]12CC=CC[C@]1(C)[C@@]1([H])CC[C@]3(C)[C@@H](C(C)=O)CC[C@@]3([H])[C@]1([H])CC2", "alpha"),
]


if __name__ == "__main__":
    rows: list[tuple[str, str, str]] = []
    for name, smiles_or_inchi, expected in EXAMPLES:
        mol = smiles_or_inchi_to_mol(smiles_or_inchi)
        if mol is None:
            rows.append((name, expected, "parse_failed"))
            continue
        rows.append((name, expected, _c5_ab_label(get_steroid_positions(mol))))

    name_width = max(len(name) for name, _, _ in rows)
    col_width = max(len("expected"), max(len(value) for _, expected, predicted in rows for value in (expected, predicted)))
    print(f"{'molecule':<{name_width}}  |  {'expected':<{col_width}}  |  predicted")
    print(f"{'-' * name_width}--+--{'-' * col_width}--+----------")
    for name, expected, predicted in rows:
        print(f"{name:<{name_width}}  |  {expected:<{col_width}}  |  {predicted}")
