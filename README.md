# GNINA Docking Web App

A FastAPI-based web application for structure-based molecular docking using [GNINA](https://github.com/gnina/gnina), with real-time progress tracking, optional post-processing analyses, and PyMOL session generation.

## Features

- **Flexible ligand input**: SMILES strings or SDF files (2D or 3D)
- **Automatic 2D→3D conversion**: OpenBabel-based 3D coordinate generation and protonation at target pH
- **Dual-GPU docking**: Load-balanced across two GPUs via `CUDA_VISIBLE_DEVICES`
- **Real-time progress**: WebSocket-based live updates during docking
- **Optional post-processing** (per pose, written as SDF fields):
  - `MCS_RMSD` — MCS-aligned RMSD vs reference ligand (RDKit, heavy atoms only)
  - `Shape_Sim` — 3D shape Tanimoto similarity vs reference ligand (RDKit)
  - `Ref_Sim` — 2D Morgan ECFP4 Tanimoto similarity vs reference ligand (RDKit)
  - `PB_Flags` — [PoseBusters](https://github.com/maabuu/posebusters) failure count (`config='mol'`)
- **PyMOL session**: Headless PyMOL generates a `.pse` file with protein rainbow cartoon, binding site surface, reference ligand (green sticks), and docked poses
- **Named sessions**: User-defined session name propagated to output SDF, PSE, and ZIP filenames
- **DataWarrior-compatible SDF output**: Correct protonation states (COO⁻, NH₃⁺), proper block formatting, DockingRank field

## Requirements

### External tools
- [GNINA](https://github.com/gnina/gnina) ≥ 1.3
- [OpenBabel](https://openbabel.org) ≥ 3.1 (with tetrazole phmodel fix — see below)
- [PyMOL](https://pymol.org) (open-source or commercial, headless-capable)

### Python environment
```bash
conda create -n gnina_webapp python=3.10
conda activate gnina_webapp
conda install -c conda-forge rdkit openbabel
pip install -r requirements.txt
pip install posebusters  # optional, for PB_Flags
```

## Usage

```bash
conda activate gnina_webapp
uvicorn gnina_webapp:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in your browser.

### Inputs
| Field | Description |
|---|---|
| Receptor (PDB) | Protein structure for docking |
| Reference ligand (SDF) | Defines the binding site (autobox); used as reference for post-processing metrics |
| Ligands (SDF or SMILES) | Molecules to dock |
| Session name | Prefix for output filenames |
| LigPrep pH | Target pH for OpenBabel protonation |
| Poses per ligand | Number of docked poses to generate |
| Exhaustiveness | GNINA search exhaustiveness (1–64) |
| CNN scoring | GNINA CNN scoring mode |

### Outputs
A ZIP file containing:
- `{session_name}.sdf` — all docked poses with scoring and optional post-processing fields, ready for DataWarrior
- `{session_name}.pse` — PyMOL session (if requested)

## SDF Output Fields

| Field | Description |
|---|---|
| `minimizedAffinity` | Vina-style binding affinity (kcal/mol) |
| `CNNscore` | GNINA CNN pose quality score |
| `CNNaffinity` | GNINA CNN predicted affinity |
| `DockingRank` | Pose rank within each ligand |
| `MCS_RMSD` | MCS-aligned RMSD to reference (Å) |
| `Shape_Sim` | 3D shape Tanimoto similarity to reference (0–1) |
| `Ref_Sim` | 2D ECFP4 Tanimoto similarity to reference (0–1) |
| `PB_Flags` | Number of failed PoseBusters checks |

## Configuration

Key settings near the top of `gnina_webapp.py`:

```python
GNINA_PATH = "/opt/gnina/gnina.1.3.2"
PYMOL_PATH = "/path/to/pymol"
DOCK_GPU_ID = 1        # CUDA device index for docking
N_GPU = 1              # Number of GPUs to use
```

## Notes

### OpenBabel tetrazole fix
OpenBabel ≥ 3.1 can double-protonate tetrazoles. Apply this fix to
`/usr/local/share/openbabel/3.1.1/phmodel.txt` (adjust path for your install):

```
TRANSFORM c1[nH:1]nnn1 >> c1[n-:1]nnn1  4.89
TRANSFORM c1n[nH:1]nn1 >> c1n[n-:1]nn1  4.89
```

### GNINA GPU selection
The `--device` flag is ignored in GNINA 1.3.2. GPU selection is done via
`CUDA_VISIBLE_DEVICES` on the subprocess instead.
