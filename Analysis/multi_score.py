"""
Pairwise TM-score + lDDT calculator for .gro simulation files in the output of an entire run
Uses the vacuum simulations before each exlosion 
The strcuture of the input directory and output h5 file is found below

Folder structure expected:
    model_folder/
        protein_<name>/
            sim_00/gro.gro
            sim_01/gro.gro
            ...
            sim_99/gro.gro

Output HDF5 layout:
    scores.h5
    ├── protein_A_vs_protein_B/
    │   ├── tm_score          shape (100, 100)  float32
    │   └── lddt              shape (100, 100)  float32
    ├── protein_A_vs_protein_C/
    │   └── ...
    └── ...  (all ordered pairs, i < j)


"""

# ──────────────────────────────────────────────
# CONFIG  –  edit these to fit your desired protein
# ──────────────────────────────────────────────
MODEL_FOLDER   = "resultat/rfah/output_data"   # root folder containing protein_* subdirs
N_SIMS         = 100              # sims per protein  (sim_00 … sim_99)
GRO_FILENAME   = "gro.gro"        # filename inside each sim_XX folder
OUTPUT_H5      = "scores_rfah.h5"      # output HDF5 file
N_cores = 16
INCLUDE_SELF  = True
# ──────────────────────────────────────────────

import os
import re
import tempfile
import warnings
import numpy as np
import h5py
from pathlib import Path
import MDAnalysis as mda
from tmtools.io import get_structure, get_residue_data
from tmtools import tm_align
from Bio.PDB import PDBParser
from concurrent.futures import ProcessPoolExecutor, as_completed
# silence noisy Bio / MDAnalysis warnings
warnings.filterwarnings("ignore")



# ── .gro → .pdb conversion ───────────────────
def gro_to_pdb(gro_path: str, pdb_path: str) -> None:
    """Convert a GROMACS .gro file to PDB using MDAnalysis."""
    u = mda.Universe(gro_path)
    with mda.Writer(pdb_path, multiframe=False) as w:
        w.write(u.atoms)


# ── TM-score ─────────────────────────────────
def calc_tm_score(pdb1: str, pdb2: str) -> float:
    try:
        s1 = get_structure(pdb1)
        s2 = get_structure(pdb2)
        c1 = next(s1.get_chains())
        c2 = next(s2.get_chains())
        coords1, seq1 = get_residue_data(c1)
        coords2, seq2 = get_residue_data(c2)
        res = tm_align(coords1, coords2, seq1, seq2)
        return float(res.tm_norm_chain2)
    except Exception:
        return float("nan")


# ── lDDT helpers ─────────────────────────────
def get_ca_coords(pdb_file: str) -> dict:
    """Return {res_seq_num: CA_coord_array} for the first chain."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", pdb_file)
    coords = {}
    model = structure[0]
    chain = list(model.get_chains())[0]
    for residue in chain:
        if "CA" in residue:
            res_num = residue.get_id()[1]
            coords[res_num] = residue["CA"].get_coord()
    return coords


def fast_lddt(
    coords1: dict,
    coords2: dict,
    cutoff: float = 15.0,
    thresholds: list = [0.5, 1.0, 2.0, 4.0],
) -> float:
    common = sorted(set(coords1.keys()) & set(coords2.keys()))
    if not common:
        return float("nan")

    c1 = np.array([coords1[r] for r in common])
    c2 = np.array([coords2[r] for r in common])

    d1 = np.sqrt(np.sum((c1[:, None, :] - c1[None, :, :]) ** 2, axis=-1))
    d2 = np.sqrt(np.sum((c2[:, None, :] - c2[None, :, :]) ** 2, axis=-1))

    mask = (d1 < cutoff) & (d1 > 0)
    diff = np.abs(d1 - d2)
    n_total = np.sum(mask)
    if n_total == 0:
        return float("nan")

    scores = [np.sum((diff < t) & mask) / n_total for t in thresholds]
    return float(np.mean(scores))


def calc_lddt(pdb1: str, pdb2: str) -> float:
    try:
        return fast_lddt(get_ca_coords(pdb1), get_ca_coords(pdb2))
    except Exception:
        return float("nan")


# ── Discover proteins ─────────────────────────
def discover_proteins(model_folder: str) -> list[tuple[str, list[str]]]:
    root = Path(os.path.abspath(model_folder))  # ← gör om till absolut sökväg
    protein_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir()]
    )
    if not protein_dirs:
        raise FileNotFoundError(
            f"No 'protein_*' subdirectories found in {root}"  # ← visar absolut sökväg i felet
        )
    return protein_dirs


def sim_gro_path(protein_dir: Path, sim_idx: int, gro_filename: str) -> Path:
    sim_name = f"sim_{sim_idx:02d}"
    return protein_dir / sim_name / gro_filename


# ── Cache all proteins as PDB files in a temp dir ──
def cache_pdbs(protein_dirs: list, n_sims: int, gro_filename: str, tmp_dir: str):
    """
    Convert all .gro files to .pdb and cache them.
    Returns dict: {protein_name: [pdb_path_0, pdb_path_1, ...]}
    """
    cache = {}
    for pdir in protein_dirs:
        pname = pdir.name
        pdb_list = []
        print(f"  Converting {pname} …", flush=True)
        for s in range(n_sims):
            gro = sim_gro_path(pdir, s, gro_filename)
            if not gro.exists():
                print(f"    WARNING: missing {gro}")
                pdb_list.append(None)
                continue
            pdb_out = os.path.join(tmp_dir, f"{pname}_sim{s:03d}.pdb")
            gro_to_pdb(str(gro), pdb_out)
            pdb_list.append(pdb_out)
        cache[pname] = pdb_list
    return cache


# ── Main ──────────────────────────────────────


# Worker: beräknar hela (n_sims x n_sims) matrisen för ett par
def compute_pair(args):
    pA, pB, pdbs_A, pdbs_B, n_sims, self_compare = args
    tm_matrix   = np.full((n_sims, n_sims), np.nan, dtype=np.float32)
    lddt_matrix = np.full((n_sims, n_sims), np.nan, dtype=np.float32)

    for si in range(n_sims):
        if pdbs_A[si] is None:
            continue
        for sj in range(n_sims):
            if pdbs_B[sj] is None:
                continue
            # vid självjämförelse: hoppa över exakt samma sim (trivial 1.0)
            if self_compare and si == sj:
                continue
            tm_matrix[si, sj]   = calc_tm_score(pdbs_A[si], pdbs_B[sj])
            lddt_matrix[si, sj] = calc_lddt(pdbs_A[si], pdbs_B[sj])

    return pA, pB, tm_matrix, lddt_matrix


def main():
    print("=== Pairwise GRO score calculator ===\n")
    protein_dirs = discover_proteins(MODEL_FOLDER)
    print(f"Found {len(protein_dirs)} protein(s): {[d.name for d in protein_dirs]}\n")

    with tempfile.TemporaryDirectory() as tmp_dir:
        print("Step 1/2  –  Converting .gro files to .pdb …")
        cache = cache_pdbs(protein_dirs, N_SIMS, GRO_FILENAME, tmp_dir)
        print()

        protein_names = [d.name for d in protein_dirs]
        n_proteins = len(protein_names)

        # Bygg alla par (i < j) + ev. självpar (i == j)
        pairs = []
        for i in range(n_proteins):
            for j in range(i, n_proteins):   # ← i istället för i+1 ger självpar
                if i == j and not INCLUDE_SELF:
                    continue
                pA, pB = protein_names[i], protein_names[j]
                self_compare = (i == j)
                pairs.append((pA, pB, cache[pA], cache[pB], N_SIMS, self_compare))

        print(f"Step 2/2  –  Computing {len(pairs)} pair(s) with {N_cores} workers …\n")

        with h5py.File(OUTPUT_H5, "w") as h5:
            with ProcessPoolExecutor(max_workers=N_cores) as executor:
                futures = {executor.submit(compute_pair, p): p[0:2] for p in pairs}
                for future in as_completed(futures):
                    pA, pB, tm_matrix, lddt_matrix = future.result()
                    key = f"{pA}_vs_{pB}"
                    print(f"  ✓ {key}", flush=True)
                    grp = h5.create_group(key)
                    grp.create_dataset("tm_score", data=tm_matrix,   compression="gzip")
                    grp.create_dataset("lddt",     data=lddt_matrix, compression="gzip")
                    grp.attrs["protein_A"] = pA
                    grp.attrs["protein_B"] = pB
                    grp.attrs["rows"]      = f"{pA} sim index"
                    grp.attrs["cols"]      = f"{pB} sim index"

    
                    

    print(f"\nDone!  Results saved to: {OUTPUT_H5}")
    print("\nHDF5 structure:")
    with h5py.File(OUTPUT_H5, "r") as h5:
        for key in h5.keys():
            grp = h5[key]
            print(f"  /{key}/tm_score   shape={grp['tm_score'].shape}")
            print(f"  /{key}/lddt       shape={grp['lddt'].shape}")


if __name__ == "__main__":
    main()