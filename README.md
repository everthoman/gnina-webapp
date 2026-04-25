# GNINA Docking Web App

A FastAPI-based web application for structure-based molecular docking using [GNINA](https://github.com/gnina/gnina), with real-time progress tracking, optional post-processing analyses, and PyMOL session generation.

## Features

- **Flexible ligand input**: SMILES strings or SDF files (2D or 3D); SMILES lines that exceed the input box width are flagged inline and blocked at submission to prevent identifier-mismatch errors from browser line-wrapping
- **Automatic 2D→3D conversion**: OpenBabel-based 3D coordinate generation and protonation at target pH
- **Dual-GPU docking**: Load-balanced across two GPUs via `CUDA_VISIBLE_DEVICES`
- **Real-time progress**: WebSocket-based live updates during docking
- **Optional post-processing** (per pose, written as SDF fields):
  - `MCS_RMSD` — MCS-aligned RMSD vs reference ligand (RDKit, heavy atoms only)
  - `Shape_Sim` — 3D shape Tanimoto similarity vs reference ligand (RDKit)
  - `Ref_Sim` — 2D Morgan ECFP4 Tanimoto similarity vs reference ligand (RDKit)
  - `PLIF_Sim` — Protein-Ligand Interaction Fingerprint Tanimoto similarity vs reference ligand (ODDT)
  - `PB_Flags` — [PoseBusters](https://github.com/maabuu/posebusters) failure count (`config='mol'`)
- **Protein preparation**: Optional integrated pipeline — fetch by PDB ID or upload, select chains and reference ligand, auto-populates docking inputs. Pipeline: PDBFixer repair → PDBFixer protonation → ASN/GLN/HIS rotamer optimisation → OpenMM minimization (heavy atoms frozen, H positions optimised to convergence). Unchecking a chain in the UI hides its HETATM groups automatically. Prepared receptor and reference ligand can be saved to browser storage for quick reuse, or downloaded to disk.
- **PyMOL session**: Headless PyMOL generates a `.pse` file with protein rainbow cartoon, pocket residues shown as full lines with residue labels, surface on atoms within 5 Å of poses, reference ligand (green sticks), and docked poses. Each unique ligand (identified by its SMILES identifier or SDF title) becomes a separate PyMOL object; multiple poses become states ordered by the selected sort metric (best pose = state 1).
- **Named sessions**: User-defined session name propagated to output SDF, PSE, and ZIP filenames
- **DataWarrior-compatible SDF output**: Correct protonation states (COO⁻, NH₃⁺), proper block formatting, DockingRank field

## Requirements

### External tools
- [GNINA](https://github.com/gnina/gnina) ≥ 1.3
- [OpenBabel](https://openbabel.org) ≥ 3.1 (with tetrazole phmodel fix — see below)
- [PyMOL](https://pymol.org) (open-source or commercial, headless-capable)

### Python environments

**Main app** (`gnina_webapp` env):
```bash
conda create -n gnina_webapp python=3.10
conda activate gnina_webapp
conda install -c conda-forge rdkit openbabel
pip install -r requirements.txt
pip install posebusters  # optional, for PB_Flags
```

**Protein preparation** (`openmmdl` env — only needed for the Protein Preparation feature):
```bash
conda create -n openmmdl python=3.10
conda activate openmmdl
conda install -c conda-forge biopython pdbfixer openmm
```
The protein preparation pipeline (`protprep.py`) is called as a subprocess using the `openmmdl` Python interpreter. The app auto-discovers it via `conda run -n openmmdl`. Override if needed:
```bash
export OPENMMDL_PYTHON=/path/to/envs/openmmdl/bin/python
```

## Usage

```bash
conda activate gnina_webapp
python gnina_webapp.py
```

Then open `http://localhost:9000` in your browser.

### Running as a persistent service

To keep the app running across sessions and restart it automatically on reboot, use a systemd service. Create `/etc/systemd/system/gnina-webapp.service`:

```ini
[Unit]
Description=GNINA Docking Web App
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/path/to/gnina/webapp
ExecStart=/path/to/conda/envs/gnina_webapp/bin/python gnina_webapp.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then enable and start it:
```bash
sudo systemctl daemon-reload
sudo systemctl enable gnina-webapp
sudo systemctl start gnina-webapp
sudo systemctl status gnina-webapp   # check it's running
```

Logs are accessible via:
```bash
journalctl -u gnina-webapp -f
```

### Inputs
| Field | Description |
|---|---|
| Protein Preparation | Optional: upload or fetch PDB, select chains/reference ligand/cofactors, runs PDBFixer + OpenMM minimization, auto-fills receptor and reference inputs. Ligands and cofactors are identified by instance (`RESNAME/CHAIN:RESNUM`), so duplicate residue names in different chains are handled correctly. |
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
| `CNN_VS` | GNINA CNN virtual screening score |
| `DockingRank` | Per-ligand pose rank (1 = best pose for that ligand); used by PoseViewer to navigate states |
| `MCS_RMSD` | MCS-aligned RMSD to reference (Å) |
| `Shape_Sim` | 3D shape Tanimoto similarity to reference (0–1) |
| `Ref_Sim` | 2D ECFP4 Tanimoto similarity to reference (0–1) |
| `PLIF_Sim` | Protein-Ligand Interaction Fingerprint Tanimoto similarity to reference (0–1) |
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
