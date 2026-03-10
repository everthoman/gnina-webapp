#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
ProtPrep — Protein Preparation Pipeline
========================================
Inspired by the Schrödinger Protein Preparation Wizard.

Pipeline
--------
  1. Fetch      — download from RCSB (optional)
  2. Clean      — chain selection, altloc resolution, water / HETATM removal
  3. Fix        — missing residues, heavy atoms, selenomethionine → Met
  4. Cap        — ACE / NME terminus capping (optional)
  5. Protonate  — pH-aware H addition (PDB2PQR + PROPKA, or PDBFixer)
  6. Minimize   — restrained vacuum minimization (OpenMM, optional)

Outputs
-------
  <stem>_prepared.pdb    protonated structure (docking / scoring)
  <stem>_minimized.pdb   minimized structure  (MD / GB-SA, --minimize)
  <stem>_prep.log        full preparation log

Dependencies
------------
  Required:     biopython, pdbfixer, openmm
  Protonation:  pdb2pqr  (conda install -c conda-forge pdb2pqr)
  Rich output:  rich     (pip install rich)        — highly recommended
  CLI extras:   rich_argparse, argcomplete         — optional

Written by Claude Sonnet 4.6, 2026-02-25
"""

import argparse
import io as _io
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    np = None  # type: ignore
    _NUMPY = False

# ─── Optional imports ────────────────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.text import Text
    from rich import box
    from rich.rule import Rule
    _RICH = True
except ImportError:
    _RICH = False

try:
    from rich_argparse import RawDescriptionRichHelpFormatter as _HelpFmt
except ImportError:
    from argparse import RawDescriptionHelpFormatter as _HelpFmt  # type: ignore

try:
    import argcomplete
    from argcomplete.completers import FilesCompleter
    _ARGCOMPLETE = True
except ImportError:
    _ARGCOMPLETE = False

try:
    from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch
    from Bio.PDB.vectors import Vector
    _BIOPYTHON = True
except ImportError:
    _BIOPYTHON = False
    Select = object

try:
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile as OmmPDBFile
    _PDBFIXER = True
except ImportError:
    _PDBFIXER = False

try:
    import openmm
    import openmm.app as omm_app
    import openmm.unit as unit
    _OPENMM = True
except ImportError:
    _OPENMM = False


# ─── Console / logging setup ─────────────────────────────────────────────────

_log_lines: List[str] = []

if _RICH:
    console = Console(highlight=False)

    def _print(msg: str = "", style: str = ""):
        console.print(msg, style=style)
        _log_lines.append(re.sub(r'\[/?[^\]]*\]', '', msg))

    def _rule(title: str = ""):
        console.rule(f"[bold]{title}[/bold]" if title else "")
        _log_lines.append(f"{'─' * 60}  {title}")

    def _header(title: str, subtitle: str = ""):
        console.print(Panel(
            f"[bold white]{title}[/bold white]\n[dim]{subtitle}[/dim]" if subtitle else
            f"[bold white]{title}[/bold white]",
            style="bold blue", expand=False, padding=(0, 2)))
        _log_lines.append(f"\n{'═' * 60}")
        _log_lines.append(f"  {title}")
        if subtitle:
            _log_lines.append(f"  {subtitle}")
        _log_lines.append(f"{'═' * 60}")

    def _step(n: int, total: int, label: str):
        console.print(f"\n[bold cyan]┌─ Step {n}/{total}: {label}[/bold cyan]")
        _log_lines.append(f"\n[Step {n}/{total}] {label}")

    def _ok(msg: str):
        console.print(f"[green]│  ✓ {msg}[/green]")
        _log_lines.append(f"  ✓ {msg}")

    def _warn(msg: str):
        console.print(f"[yellow]│  ⚠ {msg}[/yellow]")
        _log_lines.append(f"  ⚠ {msg}")

    def _info(msg: str):
        console.print(f"[dim]│    {msg}[/dim]")
        _log_lines.append(f"    {msg}")

    def _err(msg: str):
        console.print(f"[bold red]│  ✗ {msg}[/bold red]")
        _log_lines.append(f"  ✗ {msg}")

else:
    def _print(msg: str = "", style: str = ""):
        print(msg); _log_lines.append(msg)

    def _rule(title: str = ""):
        print(f"{'─' * 60}  {title}" if title else "─" * 60)
        _log_lines.append(f"{'─' * 60}  {title}")

    def _header(title: str, subtitle: str = ""):
        bar = "=" * 60
        print(f"\n{bar}\n  {title}\n{bar}")
        _log_lines.extend([f"\n{bar}", f"  {title}", bar])

    def _step(n: int, total: int, label: str):
        msg = f"\n[Step {n}/{total}] {label}"
        print(msg); _log_lines.append(msg)

    def _ok(msg: str):   line = f"  ✓ {msg}"; print(line); _log_lines.append(line)
    def _warn(msg: str): line = f"  ⚠ {msg}"; print(line); _log_lines.append(line)
    def _info(msg: str): line = f"    {msg}";  print(line); _log_lines.append(line)
    def _err(msg: str):  line = f"  ✗ {msg}"; print(line); _log_lines.append(line)


def _fatal(msg: str):
    _err(msg)
    sys.exit(1)


def _fmt_time(s: float) -> str:
    if s < 60:   return f"{s:.1f} s"
    if s < 3600: return f"{s/60:.1f} min"
    return f"{s/3600:.1f} h"


# ─── BioPython helpers ────────────────────────────────────────────────────────

# Standard amino acid residue names
_STD_AA = {
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL',
    # protonation variants
    'HID','HIE','HIP','HSD','HSE','HSP','CYX','ASH','GLH','LYN',
    'MSE',  # selenomethionine — treated as standard
}

# Titratable residues and their typical pKa ranges
_TITRATABLE = {
    'ASP': (3.5, 4.5), 'GLU': (4.0, 5.0),
    'HIS': (6.0, 7.0), 'LYS': (10.0, 11.0),
    'ARG': (12.0, 13.5), 'TYR': (9.5, 10.5),
    'CYS': (8.0, 9.5),
}

# Disulfide bond SG-SG distance threshold (Å)
_SSBOND_DIST = 2.5

# Backbone heavy atoms — restrained during minimization by default
_BACKBONE_ATOMS = frozenset({'N', 'CA', 'C', 'O', 'OXT'})

# Van der Waals radii (nm) used for the frozen-ligand repulsion model.
# Values are approximate Bondi radii; unmapped elements fall back to carbon.
_VDW_RADII_NM: Dict[str, float] = {
    'C':  0.170, 'N':  0.155, 'O':  0.152, 'S':  0.180,
    'P':  0.180, 'F':  0.147, 'CL': 0.175, 'BR': 0.185,
    'I':  0.198, 'ZN': 0.139, 'MG': 0.130, 'CA': 0.197,
    'B':  0.191,
    'FE': 0.156, 'MN': 0.161, 'CU': 0.140, 'NA': 0.227,
}
_DEFAULT_VDW_NM = 0.170

def _element_vdw(element: str) -> float:
    return _VDW_RADII_NM.get(element.upper(), _DEFAULT_VDW_NM)


def _rotate_around_axis(pos, p1, p2):
    """Rotate *pos* (numpy array) by 180° around the axis p1→p2.

    Uses the 180° special case of Rodrigues' rotation formula:
        R₁₈₀(v) = 2(v·n̂)n̂ − v
    where v = pos − p1 and n̂ = normalise(p2 − p1).
    """
    v = pos - p1
    n = p2 - p1
    n = n / np.linalg.norm(n)
    return p1 + 2.0 * np.dot(v, n) * n - v


class _ChainSelector(Select):
    """Keep selected chains, resolve altlocs, remove waters and unwanted HETATM.

    Altloc resolution strategy:
      • Prefer altloc ' ' (no disorder) or 'A'.
      • If only higher-letter conformers exist for an atom (e.g. only 'B'),
        accept the one present rather than dropping the atom entirely.
    A pre-scan of the full structure is used to determine which altlocs exist
    for each (chain, residue, atom_name) triple.
    """

    def __init__(self, chains=None, keep_resnames=None, structure=None):
        self.chains = set(chains) if chains else None
        self.keep_res = {r.strip().upper() for r in (keep_resnames or [])}
        # (chain_id, res_full_id, atom_name) → set of altloc chars present
        self._atom_altlocs: Dict[tuple, set] = {}
        if structure is not None:
            self._scan_altlocs(structure)

    def _scan_altlocs(self, structure):
        """Pre-scan: record every altloc that exists for every atom."""
        for model in structure:
            for chain in model:
                cid = chain.get_id()
                for residue in chain:
                    rid = residue.get_id()
                    for atom in residue.get_unpacked_list():
                        key = (cid, rid, atom.get_name())
                        self._atom_altlocs.setdefault(key, set()).add(
                            atom.get_altloc()
                        )

    def accept_chain(self, chain):
        return self.chains is None or chain.get_id() in self.chains

    def accept_residue(self, residue):
        hetflag, _, _ = residue.get_id()
        if hetflag == ' ':   return True
        if hetflag == 'W':   return False
        return residue.get_resname().strip().upper() in self.keep_res

    def accept_atom(self, atom):
        altloc = atom.get_altloc()
        if altloc in (' ', 'A'):
            return True
        # Non-A altloc: accept only when no preferred conformer (' ' or 'A')
        # exists for this specific atom, so we never drop an atom entirely.
        if not self._atom_altlocs:
            return False
        key = (
            atom.get_parent().get_parent().get_id(),
            atom.get_parent().get_id(),
            atom.get_name(),
        )
        existing = self._atom_altlocs.get(key, set())
        return not (existing & {' ', 'A'})


def _inspect(pdb_path: Path) -> dict:
    """Extract structural statistics from a PDB file."""
    parser = PDBParser(QUIET=True)
    s = parser.get_structure('p', str(pdb_path))[0]

    chains = list(s.get_chains())
    residues = list(s.get_residues())
    atoms = list(s.get_atoms())

    std_res = [r for r in residues if r.get_id()[0] == ' ']
    het_res = [r for r in residues if r.get_id()[0] not in (' ', 'W')]
    water    = [r for r in residues if r.get_id()[0] == 'W']
    mse      = [r for r in std_res   if r.get_resname().strip() == 'MSE']
    altlocs  = [a for a in atoms if a.get_altloc() not in (' ', 'A')]

    # Chain-level residue counts
    chain_info = {}
    for ch in chains:
        std = [r for r in ch.get_residues() if r.get_id()[0] == ' ']
        chain_info[ch.get_id()] = len(std)

    # Disulfide bond detection
    cys_sg = []
    for r in std_res:
        if r.get_resname().strip() in ('CYS', 'CYX') and 'SG' in r:
            cys_sg.append(r['SG'])
    ssbonds = []
    if len(cys_sg) >= 2:
        ns = NeighborSearch(cys_sg)
        seen = set()
        for sg in cys_sg:
            neighbors = ns.search(sg.get_vector().get_array(), _SSBOND_DIST, 'A')
            for nb in neighbors:
                if nb is not sg:
                    pair = tuple(sorted([id(sg), id(nb)]))
                    if pair not in seen:
                        seen.add(pair)
                        r1 = sg.get_parent()
                        r2 = nb.get_parent()
                        ssbonds.append((
                            r1.get_parent().get_id(),
                            r1.get_id()[1],
                            r2.get_parent().get_id(),
                            r2.get_id()[1],
                        ))

    # Titratable residue count per type
    titr_counts: Dict[str, int] = {}
    for r in std_res:
        name = r.get_resname().strip()
        if name in _TITRATABLE:
            titr_counts[name] = titr_counts.get(name, 0) + 1

    het_groups = {}
    for r in het_res:
        name = r.get_resname().strip()
        ch   = r.get_parent().get_id()
        seqid = r.get_id()[1]
        het_groups.setdefault(name, []).append(f"{ch}:{seqid}")

    return {
        'chains':       [c.get_id() for c in chains],
        'chain_info':   chain_info,
        'n_std':        len(std_res),
        'n_water':      len(water),
        'n_het':        len(het_res),
        'n_mse':        len(mse),
        'het_groups':   het_groups,
        'n_altloc':     len(altlocs),
        'ssbonds':      ssbonds,
        'titr_counts':  titr_counts,
        'n_atoms':      len(atoms),
    }


# ─── Gap handling ─────────────────────────────────────────────────────────────

def _insert_ter_at_gaps(pdb_in: Path, pdb_out: Path, gap_threshold: int = 2,
                        rename_nterm_h: bool = False) -> int:
    """Insert TER records where residue numbers jump within the same chain.

    Without TER records at sequence gaps, pdb2pqr and OpenMM/PDBFixer treat
    structurally disconnected segments as covalently bonded across the gap.

    When *rename_nterm_h* is True the backbone H atom of the first residue
    immediately after each gap is renamed to H1.  pdb2pqr protonates gap
    N-terminal residues as mid-chain (outputting 'H' instead of 'H1/H2/H3');
    renaming H→H1 lets OpenMM's addHydrogens() recognise the residue as
    N-terminal and supply the missing H2/H3 with idealized geometry.
    """
    lines = pdb_in.read_text().splitlines(keepends=True)
    out_lines = []
    prev_chain: Optional[str] = None
    prev_resseq: Optional[int] = None
    n_inserted = 0
    _nterm_chain: Optional[str] = None   # chain of gap-N-terminal residue
    _nterm_resseq: Optional[int] = None  # resseq of gap-N-terminal residue

    for line in lines:
        record = line[:6].strip()
        if record in ('ATOM', 'HETATM'):
            chain = line[21]
            try:
                resseq = int(line[22:26])
            except ValueError:
                out_lines.append(line)
                continue
            if (prev_chain is not None
                    and chain == prev_chain
                    and resseq - prev_resseq >= gap_threshold):
                out_lines.append('TER\n')
                n_inserted += 1
                if rename_nterm_h:
                    _nterm_chain = chain
                    _nterm_resseq = resseq
            # Rename backbone H → H1 on the first residue after a gap
            if (rename_nterm_h
                    and chain == _nterm_chain
                    and resseq == _nterm_resseq
                    and record == 'ATOM'
                    and line[12:16].strip() == 'H'):
                line = line[:12] + ' H1 ' + line[16:]
                _nterm_chain = _nterm_resseq = None  # one rename per gap
            prev_chain = chain
            prev_resseq = resseq
        out_lines.append(line)

    pdb_out.write_text(''.join(out_lines))
    return n_inserted


# ─── Pipeline steps ───────────────────────────────────────────────────────────

def step_fetch(pdb_id: str, output_path: Path, assembly: Optional[int] = None):
    """Download a PDB entry from RCSB.

    When *assembly* is given (e.g. 1), the biological assembly PDB is fetched.
    RCSB provides assemblies in PDB format for most entries; for those that are
    CIF-only the mmCIF file is downloaded and converted to PDB via BioPython.
    """
    pdb_id = pdb_id.upper()

    if assembly:
        pdb_url = (f'https://files.rcsb.org/download/'
                   f'{pdb_id}-assembly{assembly}.pdb')
        cif_url = (f'https://files.rcsb.org/download/'
                   f'{pdb_id}-assembly{assembly}.cif')
        _info(f"Biological assembly: {assembly}")
        _info(f"URL (PDB): {pdb_url}")
        try:
            urllib.request.urlretrieve(pdb_url, str(output_path))
            return
        except urllib.error.HTTPError as e:
            if e.code != 404:
                _fatal(f"Failed to fetch {pdb_id} assembly {assembly}: "
                       f"HTTP {e.code}")
            _info("PDB format unavailable for this assembly — trying mmCIF …")

        # Fallback: download mmCIF and convert to PDB with BioPython
        _info(f"URL (mmCIF): {cif_url}")
        cif_tmp = output_path.with_suffix('.cif')
        try:
            urllib.request.urlretrieve(cif_url, str(cif_tmp))
        except urllib.error.HTTPError as e:
            _fatal(f"Failed to fetch {pdb_id} assembly {assembly}: HTTP {e.code}")
        except urllib.error.URLError as e:
            _fatal(f"Network error: {e.reason}")

        if not _BIOPYTHON:
            _fatal("BioPython is required to convert mmCIF → PDB. "
                   "pip install biopython")
        try:
            from Bio.PDB import MMCIFParser as _CIFParser
            cif_struct = _CIFParser(QUIET=True).get_structure(pdb_id, str(cif_tmp))

            # PDB format allows only single-character chain IDs.  Biological
            # assemblies often have multi-char IDs like 'A-2'; remap them.
            _CHAIN_CHARS = (
                'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                'abcdefghijklmnopqrstuvwxyz'
                '0123456789'
            )
            chain_map: dict = {}
            for model in cif_struct:
                # Collect IDs already valid (single-char) — reserved.
                used = {ch.id for ch in model if len(ch.id) == 1}
                available = [c for c in _CHAIN_CHARS if c not in used]
                avail_iter = iter(available)
                for chain in model:
                    cid = chain.id
                    if len(cid) > 1:
                        if cid not in chain_map:
                            try:
                                chain_map[cid] = next(avail_iter)
                            except StopIteration:
                                _fatal("Too many chains to remap to single-char PDB IDs")
                        chain.id = chain_map[cid]
            if chain_map:
                _info(f"Remapped {len(chain_map)} long chain ID(s) to single chars: "
                      + ', '.join(f'{k}→{v}' for k, v in chain_map.items()))

            bio_io = PDBIO()
            bio_io.set_structure(cif_struct)
            bio_io.save(str(output_path))
            cif_tmp.unlink(missing_ok=True)
            _info("mmCIF → PDB conversion done")
        except Exception as ex:
            _fatal(f"mmCIF → PDB conversion failed: {ex}")
    else:
        url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
        _info(f"URL: {url}")
        try:
            urllib.request.urlretrieve(url, str(output_path))
        except urllib.error.HTTPError as e:
            _fatal(f"Failed to fetch {pdb_id}: HTTP {e.code}")
        except urllib.error.URLError as e:
            _fatal(f"Network error: {e.reason}")


def step_clean(input_pdb: Path, output_pdb: Path,
               chains=None, keep_het=None) -> dict:
    if not _BIOPYTHON:
        _fatal("BioPython is required:  pip install biopython")

    info = _inspect(input_pdb)

    _info(f"Chains found:        {', '.join(info['chains'])}")
    for ch, n in info['chain_info'].items():
        _info(f"  Chain {ch}:          {n} residues")
    _info(f"Total standard res:  {info['n_std']}")
    _info(f"Water molecules:     {info['n_water']}")
    _info(f"HETATM groups:       {info['n_het']}")

    if info['het_groups']:
        for name, locs in info['het_groups'].items():
            kept = name.upper() in {h.upper() for h in (keep_het or [])}
            tag = '[keep]' if kept else '[remove]'
            _info(f"  {name:<6} {tag:<8} at {', '.join(locs[:4])}"
                  + (' …' if len(locs) > 4 else ''))

    if info['n_mse']:
        _warn(f"Selenomethionine (MSE): {info['n_mse']} residues — will be converted to MET by PDBFixer")

    if info['n_altloc']:
        _warn(f"Alternate locations: {info['n_altloc']} atoms — "
              "keeping conformer A (fallback: highest-occupancy conformer)")

    if info['ssbonds']:
        _ok(f"Disulfide bonds detected: {len(info['ssbonds'])}")
        for c1, r1, c2, r2 in info['ssbonds']:
            _info(f"  CYS {c1}:{r1} — CYS {c2}:{r2}")
    else:
        _info("No disulfide bonds detected")

    if info['titr_counts']:
        _info("Titratable residues:")
        for res, count in sorted(info['titr_counts'].items()):
            lo, hi = _TITRATABLE[res]
            _info(f"  {res}: {count}  (typical pKa {lo}–{hi})")

    if chains:
        missing = [c for c in chains if c not in info['chains']]
        if missing:
            _fatal(f"Chain(s) not found: {', '.join(missing)}. "
                   f"Available: {', '.join(info['chains'])}")

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('p', str(input_pdb))
    pdb_io = PDBIO()
    pdb_io.set_structure(struct)
    pdb_io.save(str(output_pdb),
                _ChainSelector(chains=chains, keep_resnames=keep_het,
                               structure=struct))
    return info


def step_fix(input_pdb: Path, output_pdb: Path,
             convert_mse: bool = True,
             replace_nonstandard: bool = True) -> dict:
    if not _PDBFIXER:
        _fatal("PDBFixer is required:  pip install pdbfixer")

    fixer = PDBFixer(filename=str(input_pdb))

    # MSE → MET is handled automatically by PDBFixer's replaceNonstandardResidues
    fixer.findMissingResidues()
    n_miss_res = sum(len(v) for v in fixer.missingResidues.values())

    fixer.findNonstandardResidues()
    nonstandard = [(r.name, s) for r, s in fixer.nonstandardResidues]
    if replace_nonstandard:
        fixer.replaceNonstandardResidues()

    fixer.findMissingAtoms()
    n_miss_atoms = sum(len(v) for v in fixer.missingAtoms.values())
    n_term_atoms = sum(len(v) for v in fixer.missingTerminals.values())

    _info(f"Missing residues:    {n_miss_res}")
    _info(f"Missing heavy atoms: {n_miss_atoms}  (+{n_term_atoms} terminal atoms)")

    if nonstandard:
        for orig, repl in nonstandard:
            _warn(f"Non-standard residue: {orig} → {repl}")

    fixer.addMissingAtoms()

    with open(str(output_pdb), 'w') as fh:
        OmmPDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)

    return {
        'n_missing_res':   n_miss_res,
        'n_missing_atoms': n_miss_atoms,
        'nonstandard':     nonstandard,
    }


def step_cap_termini(input_pdb: Path, output_pdb: Path) -> dict:
    """
    Complete missing terminal atoms at chain breaks and real termini (OXT, H, etc.).
    Uses PDBFixer's addMissingAtoms() — no ACE/NME cap residues are inserted.
    """
    if not _PDBFIXER:
        _warn("PDBFixer not available — skipping terminus capping.")
        shutil.copy(input_pdb, output_pdb)
        return {'n_caps': 0}

    fixer = PDBFixer(filename=str(input_pdb))
    fixer.findMissingResidues()
    fixer.missingResidues = {}          # don't insert gap residues
    fixer.findMissingAtoms()
    n_term = sum(len(v) for v in fixer.missingTerminals.values())
    fixer.addMissingAtoms()

    with open(str(output_pdb), 'w') as fh:
        OmmPDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)

    return {'n_caps': n_term}


def step_protonate_pdb2pqr(input_pdb: Path, output_pdb: Path,
                            ph: float, ff: str = 'AMBER') -> Tuple[bool, dict]:
    """PDB2PQR + PROPKA: assign protonation states at target pH."""
    pqr_out = output_pdb.with_suffix('.pqr')
    propka_out = output_pdb.with_suffix('.propka')

    cmd = [
        'pdb2pqr',
        '--ff', ff,
        '--titration-state-method', 'propka',
        '--with-ph', str(ph),
        '--pdb-output', str(output_pdb),
        '--keep-chain',
        '--drop-water',
        str(input_pdb),
        str(pqr_out),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=str(output_pdb.parent))
    except FileNotFoundError:
        _warn("pdb2pqr not found in PATH — falling back to PDBFixer")
        return False, {}

    if result.returncode != 0:
        _warn(f"PDB2PQR exited with code {result.returncode}")
        for line in result.stderr.strip().splitlines()[:6]:
            _info(f"  {line}")
        return False, {}

    # Parse PROPKA output for ionization report
    propka_info = _parse_propka(propka_out, ph)

    # Clean up temporary files
    for f in list(output_pdb.parent.glob('*.pqr')) + \
              list(output_pdb.parent.glob('*.propka')):
        f.unlink(missing_ok=True)

    return True, propka_info


def _parse_propka(propka_path: Path, ph: float) -> dict:
    """Parse PROPKA output to extract pKa predictions and protonation states."""
    if not propka_path.exists():
        return {}
    try:
        text = propka_path.read_text()
    except Exception:
        return {}

    results = []
    # Match lines like:  ASP  18 A    3.80  (3.80)  ...
    pattern = re.compile(
        r'^(ASP|GLU|HIS|LYS|ARG|TYR|CYS|NTR|CTR)\s+(\d+)\s+(\w)\s+([\d.]+)',
        re.MULTILINE
    )
    for m in pattern.finditer(text):
        resname, resid, chain, pka = m.groups()
        pka_val = float(pka)
        protonated = pka_val > ph  # residue is protonated if pKa > pH
        # for acids: protonated = neutral (COOH); for bases: protonated = charged (+NH3)
        results.append({
            'resname': resname, 'resid': int(resid), 'chain': chain,
            'pka': pka_val, 'protonated': protonated,
        })
    return {'propka': results}


def _report_protonation(propka_info: dict, ph: float):
    """Print a PPW-style ionization table."""
    entries = propka_info.get('propka', [])
    if not entries:
        return

    acids   = [e for e in entries if e['resname'] in ('ASP','GLU','CTR')]
    bases   = [e for e in entries if e['resname'] in ('LYS','ARG','HIS','NTR')]
    other   = [e for e in entries if e['resname'] in ('TYR','CYS')]
    special = []

    # Flag residues with unusual protonation at this pH
    for e in entries:
        rn = e['resname']
        pka = e['pka']
        # Flag if pKa is "near" pH (within 1 unit) — these could be ambiguous
        if abs(pka - ph) < 1.0 and rn not in ('NTR','CTR'):
            special.append(e)

    _info(f"Ionization report at pH {ph}:")

    if _RICH:
        tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta",
                    show_edge=False, padding=(0, 1))
        tbl.add_column("Residue", style="cyan", no_wrap=True)
        tbl.add_column("Chain")
        tbl.add_column("Seq#", justify="right")
        tbl.add_column("pKa", justify="right")
        tbl.add_column("State at pH " + str(ph))
        tbl.add_column("Note")

        for e in sorted(entries, key=lambda x: (x['chain'], x['resid'])):
            rn = e['resname']
            state, color = _protonation_label(rn, e['protonated'])
            note = "[yellow]⚠ near pH[/yellow]" if e in special else ""
            tbl.add_row(rn, e['chain'], str(e['resid']),
                        f"{e['pka']:.2f}", f"[{color}]{state}[/{color}]", note)
        console.print(tbl)
        _log_lines.append(f"  {len(entries)} titratable residues analyzed by PROPKA")
    else:
        _info(f"  {'Res':<6} {'Chain':>5} {'Seq#':>5}  {'pKa':>6}  State")
        for e in sorted(entries, key=lambda x: (x['chain'], x['resid'])):
            state, _ = _protonation_label(e['resname'], e['protonated'])
            flag = " ⚠" if e in special else ""
            _info(f"  {e['resname']:<6} {e['chain']:>5} {e['resid']:>5}  "
                  f"{e['pka']:>6.2f}  {state}{flag}")

    if special:
        _warn(f"{len(special)} residue(s) have pKa within 1 unit of pH {ph} "
              f"— protonation state may be ambiguous; consider manual inspection.")


def _protonation_label(resname: str, protonated: bool) -> Tuple[str, str]:
    """Return human-readable state label and rich color."""
    mapping = {
        ('ASP', True):  ("neutral (–COOH)",   "white"),
        ('ASP', False): ("anionic (–COO⁻)",   "green"),
        ('GLU', True):  ("neutral (–COOH)",   "white"),
        ('GLU', False): ("anionic (–COO⁻)",   "green"),
        ('HIS', True):  ("protonated (HIP+)", "yellow"),
        ('HIS', False): ("neutral (HIE/HID)", "green"),
        ('LYS', True):  ("protonated (+NH₃)", "green"),
        ('LYS', False): ("neutral (–NH₂)",    "yellow"),
        ('ARG', True):  ("protonated (+)",     "green"),
        ('ARG', False): ("neutral",            "yellow"),
        ('TYR', True):  ("neutral (–OH)",      "white"),
        ('TYR', False): ("anionic (–O⁻)",      "yellow"),
        ('CYS', True):  ("neutral (–SH)",      "white"),
        ('CYS', False): ("thiolate (–S⁻)",     "yellow"),
        ('NTR', True):  ("N-term protonated",  "green"),
        ('NTR', False): ("N-term neutral",     "white"),
        ('CTR', True):  ("C-term neutral",     "white"),
        ('CTR', False): ("C-term anionic",     "green"),
    }
    return mapping.get((resname, protonated), ("unknown", "dim"))


def step_protonate_pdbfixer(input_pdb: Path, output_pdb: Path, ph: float):
    """PDBFixer fallback: add hydrogens at target pH without PROPKA."""
    if not _PDBFIXER:
        _fatal("PDBFixer is required:  pip install pdbfixer")
    fixer = PDBFixer(filename=str(input_pdb))
    fixer.addMissingHydrogens(ph)
    with open(str(output_pdb), 'w') as fh:
        OmmPDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)


def _normalize_his_names(model) -> int:
    """Rename HIS-family residues to the correct HID/HIE/HIP variant.

    Inspects which imidazole ring hydrogen atoms are actually present:
      HD1 present, HE2 absent  →  HID  (neutral, δ-protonated)
      HE2 present, HD1 absent  →  HIE  (neutral, ε-protonated)
      both present             →  HIP  (protonated cation, +1)
      neither present          →  HIE  (default neutral; warns — may indicate
                                         a failed protonation step)

    Also handles CHARMM naming (HSD/HSE/HSP).  Returns the number of
    residues whose name was changed.
    """
    _charmm_his = frozenset({'HSD', 'HSE', 'HSP'})
    _his_variants = frozenset({'HIS', 'HID', 'HIE', 'HIP', 'HSD', 'HSE', 'HSP'})
    n_renamed = 0
    for chain in model:
        for res in chain:
            rn = res.get_resname().strip().upper()
            if rn not in _his_variants:
                continue
            use_charmm = rn in _charmm_his
            hd1 = 'HD1' in res
            he2 = 'HE2' in res
            if hd1 and he2:
                correct = 'HSP' if use_charmm else 'HIP'
            elif hd1:
                correct = 'HSD' if use_charmm else 'HID'
            elif he2:
                correct = 'HSE' if use_charmm else 'HIE'
            else:
                # No imidazole H found.  A bare HIS (no HD1, no HE2) causes
                # force fields and viewers to apply an ambiguous charge model,
                # producing the characteristic "+N / −N" appearance on the ring.
                # Default to HIE (ε-tautomer — statistically the more common
                # neutral form) and warn so the user can inspect manually.
                correct = 'HSE' if use_charmm else 'HIE'
                rid = res.get_id()
                _warn(f"HIS {chain.id}{rid[1]}: no imidazole H found after "
                      f"protonation — defaulting to {correct}. "
                      f"Inspect manually (metal coordination? unusual environment?).")
            if res.get_resname().strip() != correct:
                res.resname = correct
                n_renamed += 1
    return n_renamed


def step_normalize_his(input_pdb: Path, output_pdb: Path) -> int:
    """Read a protonated PDB, normalise HIS residue names, write back.

    Runs _normalize_his_names unconditionally so the correct HID/HIE/HIP
    naming is applied even when rotamer flipping is disabled (--no-flip).
    Returns the number of renamed residues, or 0 if BioPython is unavailable.
    """
    if not _BIOPYTHON:
        shutil.copy(input_pdb, output_pdb)
        return 0
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('p', str(input_pdb))
    model = structure[0]
    n = _normalize_his_names(model)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_pdb))
    return n


_HIS_VAR_RE = re.compile(
    r'^((?:ATOM  |HETATM).{11})(HID|HIE|HIP)( )(.)(.{4})',
    re.MULTILINE,
)


def _rename_his_to_his(pdb_path: Path) -> int:
    """Rename HID/HIE/HIP → HIS in a PDB file for viewer compatibility.

    The protonation state is preserved in hydrogen atom positions regardless
    of the residue name.  Returns the number of unique HIS residues renamed.
    """
    text = pdb_path.read_text()
    seen: set = set()

    def _sub(m: re.Match) -> str:
        seen.add((m.group(4), m.group(5).strip()))   # (chain, resseq)
        return m.group(1) + 'HIS' + m.group(3) + m.group(4) + m.group(5)

    new_text = _HIS_VAR_RE.sub(_sub, text)
    if new_text != text:
        pdb_path.write_text(new_text)
    return len(seen)


def _count_clashes(pdb_path: Path, cutoff: float = 1.5) -> int:
    """Count severe steric clashes (heavy-atom pairs closer than cutoff Å)."""
    if not _BIOPYTHON:
        return -1
    try:
        parser = PDBParser(QUIET=True)
        s = parser.get_structure('p', str(pdb_path))[0]
        heavy = [a for a in s.get_atoms()
                 if a.element not in (None, 'H') and a.get_altloc() in (' ', 'A')]
        if not heavy:
            return 0
        ns = NeighborSearch(heavy)
        clashes = 0
        seen = set()
        for atom in heavy:
            nbs = ns.search(atom.get_vector().get_array(), cutoff, 'A')
            for nb in nbs:
                if nb is atom:
                    continue
                pair = tuple(sorted([id(atom), id(nb)]))
                if pair in seen:
                    continue
                seen.add(pair)
                r1 = atom.get_parent()
                r2 = nb.get_parent()
                # Skip intra-residue pairs and adjacent-residue backbone bonds
                # (e.g. C(i)–N(i+1) ~1.33 Å is below the clash cutoff)
                if r1 is r2:
                    continue
                if (r1.get_parent() is r2.get_parent() and
                        abs(r1.get_id()[1] - r2.get_id()[1]) <= 1):
                    continue
                clashes += 1
        return clashes
    except Exception:
        return -1


def _split_protein_hetatm(
        pdb_path: Path, protein_out: Path) -> Tuple[List[str], List[dict]]:
    """Write a protein-only PDB (no HETATM) to *protein_out*.

    Returns
    -------
    hetatm_lines : list[str]
        Original HETATM record lines, preserved verbatim to reattach after
        minimization.
    hetatm_heavy : list[dict]
        One entry per HETATM heavy atom: {'pos': (x, y, z) in nm, 'element': str}.
        Used to build repulsive wall forces during minimization.
    """
    lines = pdb_path.read_text().splitlines(keepends=True)
    protein_lines: List[str] = []
    hetatm_lines:  List[str] = []
    hetatm_heavy:  List[dict] = []

    for line in lines:
        record = line[:6].strip()
        if record == 'HETATM':
            hetatm_lines.append(line)
            try:
                x = float(line[30:38]) / 10.0   # Å → nm
                y = float(line[38:46]) / 10.0
                z = float(line[46:54]) / 10.0
                # Element column (cols 77-78) preferred; fall back to atom name
                raw_elem = line[76:78].strip() if len(line) > 76 else ''
                if not raw_elem:
                    raw_elem = ''.join(c for c in line[12:16].strip()
                                       if c.isalpha())[:2]
                element = raw_elem.upper()
                if element and element[0] != 'H':   # heavy atoms only
                    hetatm_heavy.append({'pos': (x, y, z), 'element': element})
            except (ValueError, IndexError):
                pass
        else:
            protein_lines.append(line)

    protein_out.write_text(''.join(protein_lines))
    return hetatm_lines, hetatm_heavy


def _add_ligand_wall_forces(system, topology, hetatm_heavy: List[dict]) -> int:
    """Add one CustomExternalForce per HETATM heavy atom.

    Each force repels all protein heavy atoms from a fixed ligand-atom
    position using a purely-repulsive r⁻¹² wall:

        U = ε · (σ / max(r, r_floor))¹²

    where σ = r_vdW(protein_generic) + r_vdW(ligand_atom) and ε = 1 kJ/mol.
    The floor (0.1 nm) prevents a singularity at r → 0.

    This keeps the binding-site protein geometry consistent with the
    ligand's presence without requiring force-field parameters for the ligand.

    Wall forces are applied only to non-polar protein heavy atoms (C, S, P,
    halogens, metals).  Polar O and N atoms are excluded because they may be
    in H-bond contact with the ligand; applying a repulsive wall to them
    would break those H-bonds during minimization.

    Returns the number of wall forces added (= number of HETATM heavy atoms).
    """
    # Elements that can H-bond: excluded so contacts are not disturbed
    _POLAR = frozenset({'O', 'N'})
    protein_heavy_indices = [
        atom.index for atom in topology.atoms()
        if atom.element is not None
        and atom.element.symbol not in {'H'} | _POLAR
    ]
    for lig in hetatm_heavy:
        x0, y0, z0 = lig['pos']
        sigma = _DEFAULT_VDW_NM + _element_vdw(lig['element'])
        expr = (
            f'epsilon * (sigma / max(r, r_floor))^12; '
            f'r = sqrt((x - {x0:.6f})^2 + (y - {y0:.6f})^2 + (z - {z0:.6f})^2); '
            f'sigma = {sigma:.4f}; epsilon = 1.0; r_floor = 0.10'
        )
        wall = openmm.CustomExternalForce(expr)
        for idx in protein_heavy_indices:
            wall.addParticle(idx, [])
        system.addForce(wall)
    return len(hetatm_heavy)


def step_protonate_ligand(
        hetatm_lines: List[str], work_dir: Path,
        output_sdf: Path, ph: float) -> Tuple[bool, List[dict], List[str]]:
    """Protonate HETATM residues at target pH using OpenBabel.

    Writes a protonated SDF file (for use as a GNINA reference ligand),
    returns heavy-atom positions in nm for frozen wall forces during OpenMM
    minimization, and returns the protonated HETATM lines for H-bond scoring
    in the rotamer flip step.  Falls back to original coordinates if obabel
    is absent or fails.

    Returns (sdf_written: bool, hetatm_heavy: list[dict], prot_hetatm_lines: list[str]).
    """
    lig_pdb      = work_dir / 'ligand_raw.pdb'
    lig_prot_pdb = work_dir / 'ligand_protonated.pdb'

    text = ''.join(hetatm_lines)
    if not text.rstrip().endswith('END'):
        text += 'END\n'
    lig_pdb.write_text(text)

    # Protonate with OpenBabel (-p sets pH, coordinates unchanged)
    protonated_ok = False
    try:
        result = subprocess.run(
            ['obabel', str(lig_pdb), '-O', str(lig_prot_pdb), f'-p{ph}'],
            capture_output=True, text=True,
        )
        protonated_ok = result.returncode == 0 and lig_prot_pdb.exists()
        if not protonated_ok:
            _warn(f"OpenBabel protonation failed (code {result.returncode}); "
                  "using original HETATM coordinates for wall forces")
    except FileNotFoundError:
        _warn("obabel not found — ligand protonation skipped; "
              "install openbabel for pH-aware ligand preparation")

    source_pdb = lig_prot_pdb if protonated_ok else lig_pdb

    # Convert to SDF for GNINA
    sdf_ok = False
    try:
        result2 = subprocess.run(
            ['obabel', str(source_pdb), '-O', str(output_sdf)],
            capture_output=True, text=True,
        )
        sdf_ok = result2.returncode == 0 and output_sdf.exists()
        if not sdf_ok:
            _warn("OpenBabel SDF conversion failed; no ligand SDF written")
    except FileNotFoundError:
        pass

    # Parse heavy-atom positions (Å → nm) for frozen wall forces
    hetatm_heavy: List[dict] = []
    for line in source_pdb.read_text().splitlines():
        if line[:6].strip() != 'HETATM':
            continue
        try:
            x = float(line[30:38]) / 10.0
            y = float(line[38:46]) / 10.0
            z = float(line[46:54]) / 10.0
            raw_elem = line[76:78].strip() if len(line) > 76 else ''
            if not raw_elem:
                raw_elem = ''.join(c for c in line[12:16].strip()
                                   if c.isalpha())[:2]
            element = raw_elem.upper()
            if element and element[0] != 'H':
                hetatm_heavy.append({'pos': (x, y, z), 'element': element})
        except (ValueError, IndexError):
            pass

    # Collect protonated HETATM lines for rotamer H-bond scoring
    prot_hetatm_lines: List[str] = []
    for line in source_pdb.read_text().splitlines(keepends=True):
        if line[:6].strip() == 'HETATM':
            prot_hetatm_lines.append(line)

    return sdf_ok, hetatm_heavy, prot_hetatm_lines


def step_flip_rotamers(
        prot_pdb: Path,
        output_pdb: Path,
        hetatm_lines: Optional[List[str]] = None,
        neighbor_cutoff: float = 5.0) -> Tuple[int, int]:
    """Optimise ASN / GLN amide orientation and HIS tautomer assignment.

    For each ASN and GLN residue the amide group (O + N + attached H atoms)
    is tested in both the current orientation and rotated 180° around the
    Cα–Cβ–CG or CG–CD bond axis.  The orientation that maximises H···acceptor
    contacts (< 2.5 Å) with the surrounding environment (protein + ligand)
    is kept.

    For HIS the HID (H on Nδ) ↔ HIE (H on Nε) assignment is optimised with
    the same distance-based score.  The imidazole H is repositioned using
    ideal sp² geometry when a tautomer swap is beneficial.

    Protonation states and all other atom positions are never changed.

    Parameters
    ----------
    prot_pdb : Path
        Protonated protein-only PDB (pdb2pqr / PDBFixer output).
    output_pdb : Path
        Path for the optimised output PDB.
    hetatm_lines : list[str] | None
        Protonated HETATM lines (ligand) for including the ligand in
        H-bond scoring.
    neighbor_cutoff : float
        Search radius in Å for environment atoms (default 5 Å).

    Returns
    -------
    (n_asn_gln, n_his)
        Number of ASN/GLN and HIS residues that were actually flipped.
    """
    if not _BIOPYTHON:
        _warn("BioPython not available — skipping rotamer flip")
        shutil.copy(prot_pdb, output_pdb)
        return 0, 0
    if not _NUMPY:
        _warn("NumPy not available — skipping rotamer flip")
        shutil.copy(prot_pdb, output_pdb)
        return 0, 0

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('p', str(prot_pdb))
    model = structure[0]

    # ── Collect protonated ligand atom positions for scoring ──────────────────
    # Classify ligand atoms using the same H-bond chemistry as protein atoms:
    #   acceptors: O, N, S heavy atoms only  (not C, halogens, metals)
    #   donors:    H bonded to O/N/S         (polar H, not non-polar C-H)
    # Non-polar C-H protons and carbon/halogen heavy atoms are excluded.
    _HB_ACC_ELEMS = frozenset({'O', 'N', 'S'})
    _lig_all: List[Tuple[np.ndarray, str]] = []
    if hetatm_lines:
        for line in hetatm_lines:
            if line[:6].strip() != 'HETATM':
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                raw_elem = line[76:78].strip() if len(line) > 76 else ''
                if not raw_elem:
                    raw_elem = ''.join(c for c in line[12:16].strip()
                                       if c.isalpha())[:2]
                _lig_all.append((np.array([x, y, z]), raw_elem.upper()))
            except (ValueError, IndexError):
                pass

    # Acceptors: O/N/S ligand atoms
    lig_acc_pos: List[np.ndarray] = [p for p, e in _lig_all if e in _HB_ACC_ELEMS]
    # Donors: polar H — H/D within 1.3 Å of a ligand O/N/S (O-H ~0.96 Å, N-H ~1.01 Å)
    _lig_acc_arr = np.array([p for p, e in _lig_all if e in _HB_ACC_ELEMS]) \
                   if lig_acc_pos else np.zeros((0, 3))
    lig_don_pos: List[np.ndarray] = []
    for pos, elem in _lig_all:
        if elem.startswith('H') or elem == 'D':
            if len(_lig_acc_arr) and np.linalg.norm(_lig_acc_arr - pos, axis=1).min() < 1.3:
                lig_don_pos.append(pos)

    # ── Neighbour search over all protein atoms ───────────────────────────────
    all_atoms = list(model.get_atoms())
    ns = NeighborSearch(all_atoms)

    def _get_env(center_positions, exclude_ids, cutoff):
        """Return (acceptor_positions, donor_H_positions) arrays from environment."""
        acc: List[np.ndarray] = []
        don: List[np.ndarray] = []
        seen: set = set()
        for cp in center_positions:
            for atom in ns.search(list(cp), cutoff, 'A'):
                aid = id(atom)
                if aid in seen or aid in exclude_ids:
                    continue
                seen.add(aid)
                elem = (atom.element or '').upper().strip()
                apos = atom.coord.copy()
                if elem in ('H', 'D'):
                    don.append(apos)
                elif elem in ('O', 'N', 'S'):
                    acc.append(apos)
        for pos in lig_acc_pos:
            if any(np.linalg.norm(pos - cp) < cutoff for cp in center_positions):
                acc.append(pos.copy())
        for pos in lig_don_pos:
            if any(np.linalg.norm(pos - cp) < cutoff for cp in center_positions):
                don.append(pos.copy())
        A = np.array(acc) if acc else np.zeros((0, 3))
        D = np.array(don) if don else np.zeros((0, 3))
        return A, D

    def _hbond_score(h_positions, acc_env, o_positions, don_env, cutoff=2.5):
        """Weighted H···acceptor + acceptor···donor-H contact score."""
        score = 0.0
        c2 = cutoff * cutoff
        for h in h_positions:
            if len(acc_env):
                d2 = np.sum((acc_env - h) ** 2, axis=1)
                close = d2[d2 < c2]
                score += float(np.sum(1.0 - np.sqrt(close) / cutoff))
        for o in o_positions:
            if len(don_env):
                d2 = np.sum((don_env - o) ** 2, axis=1)
                close = d2[d2 < c2]
                score += float(np.sum(1.0 - np.sqrt(close) / cutoff))
        return score

    n_asn_gln = 0
    n_his = 0

    # ── ASN / GLN amide flips ─────────────────────────────────────────────────
    for chain in model:
        for res in chain:
            resname = res.get_resname().strip().upper()
            if resname == 'ASN':
                axis_names = ('CB', 'CG')
                flip_heavy = ['OD1', 'ND2']
                h_names    = ['HD21', 'HD22']
                acc_name   = 'OD1'
            elif resname == 'GLN':
                axis_names = ('CG', 'CD')
                flip_heavy = ['OE1', 'NE2']
                h_names    = ['HE21', 'HE22']
                acc_name   = 'OE1'
            else:
                continue

            if not all(n in res for n in list(axis_names) + flip_heavy):
                continue

            p1 = res[axis_names[0]].coord.copy()
            p2 = res[axis_names[1]].coord.copy()
            to_rotate    = [n for n in flip_heavy + h_names if n in res]
            exclude_ids  = {id(res[n]) for n in to_rotate}

            cur_h_pos = [res[n].coord.copy() for n in h_names if n in res]
            cur_o_pos = [res[acc_name].coord.copy()]

            rot: dict = {n: _rotate_around_axis(res[n].coord.copy(), p1, p2)
                         for n in to_rotate}
            rot_h_pos = [rot[n] for n in h_names if n in res]
            rot_o_pos = [rot[acc_name]]

            centers = [res[n].coord.copy() for n in flip_heavy]
            acc_env, don_env = _get_env(centers, exclude_ids, neighbor_cutoff)

            cur_score = _hbond_score(cur_h_pos, acc_env, cur_o_pos, don_env)
            rot_score = _hbond_score(rot_h_pos, acc_env, rot_o_pos, don_env)

            if rot_score > cur_score + 0.05:
                for n, new_pos in rot.items():
                    res[n].coord = new_pos
                n_asn_gln += 1

    # ── HIS tautomer (HID ↔ HIE) optimisation ────────────────────────────────
    for chain in model:
        for res in chain:
            resname = res.get_resname().strip().upper()
            if resname not in ('HIS', 'HID', 'HIE', 'HIP', 'HSP', 'HSE', 'HSD'):
                continue

            has_HD1 = 'HD1' in res
            has_HE2 = 'HE2' in res
            if has_HD1 and has_HE2:
                continue  # HIP: fully protonated
            if not has_HD1 and not has_HE2:
                continue  # no imidazole H to optimise
            if not all(n in res for n in ('ND1', 'NE2', 'CE1', 'CD2', 'CG')):
                continue

            nd1 = res['ND1'].coord.copy()
            ne2 = res['NE2'].coord.copy()
            ce1 = res['CE1'].coord.copy()
            cd2 = res['CD2'].coord.copy()
            cg  = res['CG'].coord.copy()

            def _ideal_h(n_pos, c1, c2, bond=1.01):
                """Place H at ideal sp² geometry given two bonded C atoms."""
                mid = 0.5 * (c1 + c2)
                d = n_pos - mid
                norm = np.linalg.norm(d)
                if norm < 1e-6:
                    return n_pos + np.array([0.0, 0.0, bond])
                return n_pos + (d / norm) * bond

            hd1_ideal = _ideal_h(nd1, cg,  ce1)   # H on Nδ1
            he2_ideal = _ideal_h(ne2, cd2, ce1)   # H on Nε2

            exclude_ids = set()
            if has_HD1: exclude_ids.add(id(res['HD1']))
            if has_HE2: exclude_ids.add(id(res['HE2']))
            acc_env, don_env = _get_env([nd1, ne2], exclude_ids, neighbor_cutoff)

            if has_HD1:
                cur_h, cur_acc = [hd1_ideal], [ne2]
                alt_h, alt_acc = [he2_ideal], [nd1]
            else:
                cur_h, cur_acc = [he2_ideal], [nd1]
                alt_h, alt_acc = [hd1_ideal], [ne2]

            cur_score = _hbond_score(cur_h, acc_env, cur_acc, don_env)
            alt_score = _hbond_score(alt_h, acc_env, alt_acc, don_env)

            if alt_score > cur_score + 0.05:
                from Bio.PDB.Atom import Atom as _BioAtom
                # Residue-name map: old name → new name after tautomer swap
                _his_rename = {
                    'HID': 'HIE', 'HIE': 'HID',   # AMBER naming
                    'HSD': 'HSE', 'HSE': 'HSD',   # CHARMM naming
                }
                if has_HD1:
                    res.detach_child('HD1')
                    new_h = _BioAtom('HE2', he2_ideal, 0.0, 1.0, ' ', ' HE2', None, 'H')
                    res.add(new_h)
                else:
                    res.detach_child('HE2')
                    new_h = _BioAtom('HD1', hd1_ideal, 0.0, 1.0, ' ', ' HD1', None, 'H')
                    res.add(new_h)
                # Keep residue name consistent with the H atom now present
                old_name = res.get_resname().strip()
                if old_name in _his_rename:
                    res.resname = _his_rename[old_name]
                n_his += 1

    # ── Normalise HIS residue names to match H atom assignments ──────────────
    # Re-run after any tautomer swaps to keep the residue name consistent
    # with whichever imidazole H is now present.
    n_renamed = _normalize_his_names(model)
    if n_renamed:
        _info(f"HIS residue names normalised after flip: {n_renamed}")

    # ── Write output ──────────────────────────────────────────────────────────
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_pdb))

    return n_asn_gln, n_his


def step_minimize(protein_pdb: Path, output_pdb: Path, ph: float,
                  max_iter: int = 1000, restraint_k: float = 1000.0,
                  restrain_sidechains: bool = False,
                  hetatm_heavy: Optional[List[dict]] = None,
                  has_hydrogens: bool = False) -> bool:
    """OpenMM restrained energy minimization.

    protein_pdb must contain protein atoms only (no HETATM).  When
    has_hydrogens=True the structure is assumed to be already protonated
    (e.g. by pdb2pqr) and addHydrogens() is skipped, preserving the
    assigned protonation states exactly.

    Frozen ligand wall forces are built from hetatm_heavy (a list of
    {'pos': (x,y,z) nm, 'element': str} dicts for the protonated ligand)
    when provided.  The output PDB is protein-only; the prepared ligand
    lives in its separately-written SDF.
    """
    if not _OPENMM:
        _warn("OpenMM not found — skipping minimization.")
        _info("Install:  conda install -c conda-forge openmm")
        return False

    if hetatm_heavy:
        _info(f"HETATM heavy atoms (frozen walls): {len(hetatm_heavy)}")

    # ── Re-mark sequence gaps with TER before OpenMM sees the structure ──────
    # pdb2pqr (and other upstream tools) may not re-emit mid-chain TER records
    # for same-chain segments.  Without them, PDBFixer/OpenMM creates a phantom
    # peptide bond across the gap, producing catastrophic forces during
    # minimization.  Re-running _insert_ter_at_gaps here is safe even when TER
    # records are already present (it only adds them, never removes).
    _gap_tmp = Path(tempfile.mktemp(suffix='_gapped.pdb'))
    _n_gap = _insert_ter_at_gaps(protein_pdb, _gap_tmp,
                                  rename_nterm_h=has_hydrogens)
    if _n_gap:
        _info(f"Re-marked {_n_gap} sequence gap(s) with TER before OpenMM")
    _pdb_for_fixer = _gap_tmp

    # ── Cap chain breaks before OpenMM sees the structure ────────────────────
    if _PDBFIXER:
        fixer = PDBFixer(filename=str(_pdb_for_fixer))
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        if has_hydrogens:
            # H already assigned — only let PDBFixer add missing terminal atoms
            fixer.missingAtoms = {}
        n_term = sum(len(v) for v in fixer.missingTerminals.values())
        if n_term:
            _info(f"Chain-break termini: capping {n_term} atom(s) for OpenMM")
        fixer.addMissingAtoms()
        # Use PDBFixer's topology and positions directly — writing then re-reading
        # the PDB would merge same-chain-ID gap segments (separated by TER) back
        # into one chain, causing OXT + phantom peptide bond → "bonds are different".
        _mm_topology = fixer.topology
        _mm_positions = fixer.positions
    else:
        _pdb_obj = omm_app.PDBFile(str(_pdb_for_fixer))
        _mm_topology = _pdb_obj.topology
        _mm_positions = _pdb_obj.positions

    _gap_tmp.unlink(missing_ok=True)

    ff_obj = omm_app.ForceField('amber14-all.xml')
    modeller = omm_app.Modeller(_mm_topology, _mm_positions)
    # addHydrogens is always called: when has_hydrogens=True it only adds the
    # H2/H3 atoms that pdb2pqr omitted from gap N-terminal residues (it named
    # the backbone H as 'H' rather than 'H1', so _insert_ter_at_gaps renamed
    # H→H1 above, and now addHydrogens supplies the two missing terminal H).
    # For fully-protonated non-gap residues it is a no-op.
    modeller.addHydrogens(ff_obj, pH=ph)

    try:
        system = ff_obj.createSystem(modeller.topology,
                                     nonbondedMethod=omm_app.NoCutoff)
    except Exception as e:
        _warn(f"Force field setup failed: {e}")
        _warn("Check for non-standard residue names (e.g. from pdb2pqr output).")
        _warn("Try --no-pdb2pqr or inspect the structure manually.")
        return False

    # ── Frozen ligand wall forces ────────────────────────────────────────────
    if hetatm_heavy:
        n_walls = _add_ligand_wall_forces(system, modeller.topology, hetatm_heavy)
        _info(f"Ligand wall forces added: {n_walls}")

    # ── Positional restraints on protein heavy atoms ─────────────────────────
    restraint = openmm.CustomExternalForce(
        '0.5 * k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)'
    )
    restraint.addGlobalParameter('k', restraint_k)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')

    positions = modeller.positions
    n_restrained = 0
    for atom in modeller.topology.atoms():
        if atom.element is None or atom.element.symbol == 'H':
            continue
        if restrain_sidechains or atom.name in _BACKBONE_ATOMS:
            pos = positions[atom.index].value_in_unit(unit.nanometers)
            restraint.addParticle(atom.index, pos)
            n_restrained += 1
    system.addForce(restraint)
    restrain_label = ("all heavy atoms" if restrain_sidechains
                      else "backbone atoms (N,CA,C,O)")
    _info(f"{restrain_label} restrained: {n_restrained}  "
          f"(k = {restraint_k:.0f} kJ/mol/nm²)")

    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.004 * unit.picoseconds
    )
    sim = omm_app.Simulation(modeller.topology, system, integrator)
    sim.context.setPositions(positions)

    e_before = (sim.context.getState(getEnergy=True)
                    .getPotentialEnergy()
                    .value_in_unit(unit.kilocalories_per_mole))
    _info(f"Energy before: {e_before:>12.1f} kcal/mol")

    sim.minimizeEnergy(maxIterations=max_iter)

    state = sim.context.getState(getPositions=True, getEnergy=True)
    e_after = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    _info(f"Energy after:  {e_after:>12.1f} kcal/mol  (Δ = {e_after - e_before:+.1f})")

    # ── Write protein-only output (ligand lives in its separate SDF) ─────────
    buf = _io.StringIO()
    omm_app.PDBFile.writeFile(sim.topology, state.getPositions(), buf, keepIds=True)
    output_pdb.write_text(buf.getvalue())
    return True


# ─── Final report (PPW-style) ────────────────────────────────────────────────

def _print_summary(stats: dict, args, prepared_pdb: Path,
                   minimized_pdb: Optional[Path], elapsed: float):
    _print()
    if _RICH:
        console.rule("[bold green]Preparation Complete[/bold green]")
    else:
        _rule("Preparation Complete")

    info = stats.get('info', {})
    fix  = stats.get('fix',  {})
    prot = stats.get('prot', '?')
    prot_labels = {
        'pdb2pqr+propka':    'PDB2PQR + PROPKA',
        'pdbfixer':          'PDBFixer',
        'pdbfixer_fallback': 'PDBFixer (PDB2PQR failed)',
    }

    if _RICH:
        tbl = Table(box=box.SIMPLE_HEAD, show_header=False,
                    show_edge=True, padding=(0, 1), style="dim")
        tbl.add_column("Key",   style="bold", no_wrap=True, width=28)
        tbl.add_column("Value", style="white")

        def row(k, v): tbl.add_row(k, str(v))

        row("Input file",            args.input)
        row("Chains processed",
            ', '.join(info.get('chains', ['?'])) if not args.chain
            else ', '.join(args.chain))
        row("Standard residues",     info.get('n_std', '?'))
        row("Waters removed",        info.get('n_water', '?'))

        kept_het  = [h.upper() for h in (args.keep_het or [])]
        removed_h = [h for h in info.get('het_groups', {})
                     if h.upper() not in kept_het]
        if removed_h:
            row("HETATM removed",  ', '.join(removed_h))
        if kept_het:
            row("HETATM kept",     ', '.join(kept_het))

        if info.get('n_altloc'):
            row("Altloc atoms resolved", info['n_altloc'])
        if info.get('ssbonds'):
            row("Disulfide bonds",
                ', '.join(f"{c1}{r1}–{c2}{r2}" for c1,r1,c2,r2 in info['ssbonds']))
        if fix.get('nonstandard'):
            row("Non-std res converted",
                ', '.join(f"{a}→{b}" for a,b in fix['nonstandard']))

        row("Missing residues added",  fix.get('n_missing_res',   0))
        row("Missing atoms added",     fix.get('n_missing_atoms',  0))
        row("Protonation method",      prot_labels.get(prot, prot))
        row("pH",                      args.ph)

        flips = stats.get('flips')
        if flips is not None:
            row("Rotamer flips (ASN/GLN)", flips['asn_gln'])
            row("Rotamer flips (HIS)",     flips['his'])

        clashes = stats.get('clashes_before', -1)
        clashes_after = stats.get('clashes_after', -1)
        if clashes >= 0:
            row("Clashes (before min.)", clashes)
        if clashes_after >= 0:
            row("Clashes (after min.)",  clashes_after)

        row("Output (prepared)",       str(prepared_pdb))
        if minimized_pdb and minimized_pdb.exists():
            row("Output (minimized)",  str(minimized_pdb))
        if stats.get('ligand_sdf') and Path(stats['ligand_sdf']).exists():
            row("Ligand SDF (prepared)", stats['ligand_sdf'])
        row("Elapsed",                 _fmt_time(elapsed))

        console.print(tbl)
    else:
        _rule()
        _print(f"  Input:                  {args.input}")
        _print(f"  Chains:                 {', '.join(info.get('chains', ['?']))}")
        _print(f"  Standard residues:      {info.get('n_std', '?')}")
        _print(f"  Waters removed:         {info.get('n_water', '?')}")
        _print(f"  Protonation:            {prot_labels.get(prot, prot)}  (pH {args.ph})")
        flips = stats.get('flips')
        if flips is not None:
            _print(f"  Rotamer flips:          {flips['asn_gln']} ASN/GLN,  {flips['his']} HIS")
        _print(f"  Output:                 {prepared_pdb}")
        if minimized_pdb and minimized_pdb.exists():
            _print(f"  Minimized:              {minimized_pdb}")
        if stats.get('ligand_sdf'):
            _print(f"  Ligand SDF:             {stats['ligand_sdf']}")
        _print(f"  Elapsed:                {_fmt_time(elapsed)}")
        _rule()


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    desc = """
ProtPrep — Protein Preparation Pipeline
Inspired by the Schrödinger Protein Preparation Wizard

Steps:
  1. Clean     — chain selection, altloc resolution, water/HETATM removal
  2. Fix       — missing residues/atoms, non-standard residue conversion
  3. Cap       — terminus capping for chain breaks (--cap)
  4. Protonate — pH-aware hydrogen addition (PDB2PQR + PROPKA, or PDBFixer)
  5. Minimize  — restrained vacuum energy minimization (OpenMM, --minimize)

Examples:
  %(prog)s -i protein.pdb
  %(prog)s --fetch 4HHB --chain A --ph 7.4
  %(prog)s -i complex.pdb --chain A --keep-het HEM ZN --minimize
  %(prog)s -i apo.pdb --ph 6.5 --cap --minimize --max-iter 2000
"""
    p = argparse.ArgumentParser(
        description=desc,
        formatter_class=_HelpFmt,
    )

    io = p.add_argument_group('Input / Output')
    io.add_argument('--fetch', metavar='PDBID',
                    help='Download structure from RCSB (e.g. --fetch 4HHB). '
                         'Sets --input to <PDBID>.pdb if not given.')
    io.add_argument('--assembly', type=int, default=None, metavar='N',
                    help='Biological assembly number to fetch with --fetch '
                         '(e.g. --assembly 1). Default: asymmetric unit. '
                         'PDB format is tried first; mmCIF is downloaded and '
                         'converted if PDB is unavailable.')
    inp = io.add_argument('-i', '--input', metavar='PDB',
                          help='Input PDB file (required unless --fetch is used)')
    io.add_argument('-o', '--output', metavar='PDB',
                    help='Output path for prepared structure '
                         '(default: <input>_prepared.pdb)')
    io.add_argument('--log', metavar='FILE',
                    help='Save full preparation log to file '
                         '(default: <input>_prep.log)')
    if _ARGCOMPLETE:
        inp.completer = FilesCompleter(allowednames=['.pdb', '.ent'])

    struct = p.add_argument_group('Structure')
    struct.add_argument('--chain', nargs='+', metavar='ID',
                        help='Extract one or more chains, e.g. --chain A B '
                             '(default: keep all chains)')
    struct.add_argument('--keep-het', nargs='+', metavar='RESNAME', default=[],
                        help='HETATM residue names to retain, e.g. HEM ZN MG '
                             '(waters are always removed)')

    prot = p.add_argument_group('Protonation')
    prot.add_argument('--ph', type=float, default=7.4, metavar='FLOAT',
                      help='Target pH for protonation (default: 7.4)')
    prot.add_argument('--ff', default='AMBER',
                      choices=['AMBER', 'CHARMM', 'PARSE', 'TYL06'],
                      help='Force field for PDB2PQR atom naming (default: AMBER)')
    prot.add_argument('--no-pdb2pqr', action='store_true',
                      help='Skip PDB2PQR/PROPKA; use PDBFixer for H addition')
    prot.add_argument('--no-flip', action='store_true',
                      help='Skip ASN/GLN/HIS rotamer optimisation (flips amide '
                           'groups and HIS tautomers to maximise H-bond contacts '
                           'with the environment and any kept ligand)')
    prot.add_argument('--amber-his', action='store_true',
                      help='Keep AMBER-style HID/HIE/HIP residue names in the '
                           'output PDB (default: rename back to HIS for viewer '
                           'compatibility; protonation state is preserved in '
                           'hydrogen atom positions)')

    fix_grp = p.add_argument_group('Structure Repair')
    fix_grp.add_argument('--skip-fix', action='store_true',
                         help='Skip PDBFixer repair step')
    fix_grp.add_argument('--cap', action='store_true',
                         help='Cap chain termini with ACE/NME groups '
                              '(recommended for MD simulations)')
    fix_grp.add_argument('--keep-mse', action='store_true',
                         help='Keep selenomethionine (MSE) as-is; '
                              'default is to convert to MET')

    mini = p.add_argument_group('Minimization')
    mini.add_argument('--minimize', action='store_true',
                      help='Run OpenMM vacuum minimization → <stem>_minimized.pdb')
    mini.add_argument('--max-iter', type=int, default=1000, metavar='N',
                      help='Maximum minimization iterations (default: 1000). '
                           'Use 0 to run until convergence (may be slow).')
    mini.add_argument('--restraint-k', type=float, default=1000.0, metavar='FLOAT',
                      help='Positional restraint strength in kJ/mol/nm² '
                           '(default: 1000). Higher = less movement allowed.')
    mini.add_argument('--relax-sidechains', action='store_true',
                      help='Only restrain backbone atoms (N,CA,C,O) during '
                           'minimization, letting sidechains relax freely. '
                           'Default is to restrain all heavy atoms, which '
                           'keeps the structure close to the input coordinates.')

    pipe = p.add_argument_group('Pipeline')
    pipe.add_argument('--keep-intermediates', action='store_true',
                      help='Save intermediate PDB files after each step')
    pipe.add_argument('--clash-check', action='store_true',
                      help='Count steric clashes before/after minimization '
                           '(requires BioPython; may be slow for large structures)')

    if _ARGCOMPLETE:
        argcomplete.autocomplete(p)

    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0 = time.time()

    if not args.input and not args.fetch:
        _fatal("Provide --input <file> or --fetch <PDBID>")

    if args.fetch and not args.input:
        if args.assembly:
            args.input = f'{args.fetch.upper()}-assembly{args.assembly}.pdb'
        else:
            args.input = f'{args.fetch.upper()}.pdb'

    input_pdb = Path(args.input).resolve()

    # ── Banner ────────────────────────────────────────────────────────────────
    _header("ProtPrep — Protein Preparation Pipeline",
            "Inspired by Schrödinger Protein Preparation Wizard")

    prot_method = 'PDBFixer' if args.no_pdb2pqr else 'PDB2PQR + PROPKA'
    n_steps = (2 + int(bool(args.fetch)) + int(not args.skip_fix)
               + int(args.cap) + int(not args.no_flip) + int(args.minimize))

    _print()
    _info(f"Input:         {input_pdb}")
    if args.assembly:
        _info(f"Assembly:      {args.assembly} (biological)")
    _info(f"Chain(s):      {', '.join(args.chain) if args.chain else 'all'}")
    _info(f"Keep HETATM:   {', '.join(args.keep_het) if args.keep_het else 'none'}")
    _info(f"pH:            {args.ph}")
    _info(f"Protonation:   {prot_method}")
    _info(f"Fix missing:   {'no' if args.skip_fix else 'yes'}")
    _info(f"Cap termini:   {'yes' if args.cap else 'no'}")
    if args.minimize:
        restrain_str = ("backbone only" if args.relax_sidechains else "all heavy")
        _info(f"Minimize:      yes  (k = {args.restraint_k:.0f} kJ/mol/nm², "
              f"max_iter = {args.max_iter}, restrain = {restrain_str})")
    else:
        _info("Minimize:      no")
    _info(f"Flip rotamers: {'no' if args.no_flip else 'yes  (ASN/GLN/HIS, 5 Å cutoff)'}")
    _info(f"HIS naming:    {'HID/HIE/HIP (AMBER)' if args.amber_his else 'HIS (viewer-compatible)'}")

    # ── Paths ─────────────────────────────────────────────────────────────────
    prepared_pdb = (Path(args.output).resolve() if args.output
                    else input_pdb.with_name(input_pdb.stem + '_prepared.pdb'))
    stem = prepared_pdb.stem.removesuffix('_prepared')
    minimized_pdb = prepared_pdb.with_name(stem + '_minimized.pdb')
    log_path = (Path(args.log).resolve() if args.log
                else input_pdb.with_name(stem + '_prep.log'))

    stats: dict = {}
    step_n = [0]

    def next_step(label):
        step_n[0] += 1
        _step(step_n[0], n_steps, label)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ── Fetch ─────────────────────────────────────────────────────────────
        if args.fetch:
            label = f"Downloading {args.fetch.upper()}"
            if args.assembly:
                label += f" assembly {args.assembly}"
            label += " from RCSB PDB"
            next_step(label)
            step_fetch(args.fetch, input_pdb, assembly=args.assembly)
            _ok(f"Saved  →  {input_pdb.name}")
        elif not input_pdb.exists():
            _fatal(f"File not found: {input_pdb}")
        else:
            # Still count file existence as implicit step
            pass

        # ── Step: Clean ───────────────────────────────────────────────────────
        next_step("Cleaning structure")
        clean_pdb = tmp / 'clean.pdb'
        info = step_clean(input_pdb, clean_pdb,
                          chains=args.chain, keep_het=args.keep_het)
        stats['info'] = info
        if args.keep_intermediates:
            dst = prepared_pdb.with_name(stem + '_s1_clean.pdb')
            shutil.copy(clean_pdb, dst)
            _info(f"Saved: {dst.name}")
        _ok("Cleaning done")

        # ── Step: Fix ─────────────────────────────────────────────────────────
        if not args.skip_fix:
            next_step("Repairing missing residues and atoms")
            fixed_pdb = tmp / 'fixed.pdb'
            fix_stats = step_fix(clean_pdb, fixed_pdb,
                                 replace_nonstandard=not args.keep_mse)
            stats['fix'] = fix_stats
            if args.keep_intermediates:
                dst = prepared_pdb.with_name(stem + '_s2_fixed.pdb')
                shutil.copy(fixed_pdb, dst)
                _info(f"Saved: {dst.name}")
            _ok("Repair done")
        else:
            fixed_pdb = clean_pdb
            _print()
            _warn("Skipping PDBFixer repair (--skip-fix)")

        # ── Step: Cap termini ─────────────────────────────────────────────────
        if args.cap:
            next_step("Capping chain termini")
            capped_pdb = tmp / 'capped.pdb'
            cap_stats = step_cap_termini(fixed_pdb, capped_pdb)
            fixed_pdb = capped_pdb
            stats['cap'] = cap_stats
            n_c = cap_stats.get('n_caps', 0)
            if n_c:
                _ok(f"Capped {n_c} terminal atom(s)")
            else:
                _ok("No capping needed (termini already complete)")

        # ── Gap marking: insert TER at sequence breaks ────────────────────────
        # Must happen before protonation and minimization so that pdb2pqr and
        # OpenMM/PDBFixer do not form peptide bonds across structural gaps.
        gap_pdb = tmp / 'gap_marked.pdb'
        n_gaps = _insert_ter_at_gaps(fixed_pdb, gap_pdb)
        if n_gaps:
            _info(f"Sequence gaps: {n_gaps} TER record(s) inserted at chain break(s)")
        fixed_pdb = gap_pdb

        # ── Separate protein from HETATM before protonation ───────────────────
        # pdb2pqr runs on the protein only (avoids failures on unknown ligands).
        # The ligand is protonated separately by OpenBabel so each tool handles
        # what it is designed for, and protonation states are never re-assigned.
        prot_input_pdb = tmp / 'protein_only.pdb'
        hetatm_lines, _ = _split_protein_hetatm(gap_pdb, prot_input_pdb)
        if hetatm_lines:
            n_het_rec = sum(1 for l in hetatm_lines if l[:6].strip() == 'HETATM')
            _info(f"HETATM separated: {n_het_rec} record(s) — protonated separately")

        # ── Step: Protonate (protein) ──────────────────────────────────────────
        next_step(f"Protonation at pH {args.ph}")
        prot_pdb = tmp / 'protonated.pdb'

        if args.no_pdb2pqr:
            step_protonate_pdbfixer(prot_input_pdb, prot_pdb, args.ph)
            stats['prot'] = 'pdbfixer'
        else:
            ok, propka_info = step_protonate_pdb2pqr(prot_input_pdb, prot_pdb,
                                                      args.ph, args.ff)
            if ok:
                stats['prot'] = 'pdb2pqr+propka'
                if propka_info:
                    _report_protonation(propka_info, args.ph)
            else:
                _warn("Falling back to PDBFixer hydrogen addition")
                step_protonate_pdbfixer(prot_input_pdb, prot_pdb, args.ph)
                stats['prot'] = 'pdbfixer_fallback'

        _ok("Protein protonation done  (protein only)")

        # ── Normalise HIS residue names (always, before rotamer flipping) ──────
        # pdb2pqr may leave some residues named 'HIS' (generic), and PDBFixer
        # may not rename either.  A bare HIS with no imidazole H causes force
        # fields and viewers to apply an ambiguous charge model, producing the
        # characteristic "+N / −N" appearance on the ring.  Run unconditionally
        # so the fix applies even when --no-flip is passed.
        norm_pdb = tmp / 'his_normalised.pdb'
        n_his_norm = step_normalize_his(prot_pdb, norm_pdb)
        prot_pdb = norm_pdb
        if n_his_norm:
            _info(f"HIS residue names normalised: {n_his_norm} → HID/HIE/HIP")

        # ── Protonate ligand with OpenBabel; write SDF for GNINA ──────────────
        hetatm_heavy_for_walls: List[dict] = []
        prot_hetatm_lines: List[str] = []
        if hetatm_lines:
            lig_sdf = prepared_pdb.with_name(stem + '_prepared_ligand.sdf')
            _info(f"Protonating ligand at pH {args.ph} with OpenBabel...")
            sdf_ok, hetatm_heavy_for_walls, prot_hetatm_lines = step_protonate_ligand(
                hetatm_lines, tmp, lig_sdf, args.ph)
            if sdf_ok:
                _ok(f"Ligand SDF  →  {lig_sdf.name}")
                stats['ligand_sdf'] = str(lig_sdf)
            else:
                _warn("Ligand SDF not written; wall forces use raw coordinates")

        # ── Step: Flip ASN/GLN/HIS rotamers ───────────────────────────────────
        if not args.no_flip:
            next_step("Optimising ASN/GLN/HIS rotamers (H-bond scoring)")
            flipped_pdb = tmp / 'flipped.pdb'
            n_asn_gln, n_his = step_flip_rotamers(
                prot_pdb, flipped_pdb,
                hetatm_lines=prot_hetatm_lines or None,
                neighbor_cutoff=5.0)
            prot_pdb = flipped_pdb
            stats['flips'] = {'asn_gln': n_asn_gln, 'his': n_his}
            _ok(f"Rotamer flips: {n_asn_gln} ASN/GLN,  {n_his} HIS")

        shutil.copy(prot_pdb, prepared_pdb)
        if not args.amber_his:
            _rename_his_to_his(prepared_pdb)
        _ok(f"Prepared protein  →  {prepared_pdb.name}")

        # Clash check (optional)
        if args.clash_check:
            n_clashes = _count_clashes(prepared_pdb)
            stats['clashes_before'] = n_clashes
            if n_clashes < 0:
                _warn("Clash detection unavailable")
            elif n_clashes == 0:
                _ok("No severe steric clashes detected")
            else:
                _warn(f"Steric clashes detected: {n_clashes}  "
                      "(consider --minimize to resolve)")

        # ── Step: Minimize ────────────────────────────────────────────────────
        if args.minimize:
            next_step("Energy minimization (OpenMM, ff14SB, vacuum)")
            ok = step_minimize(prot_pdb, minimized_pdb, ph=args.ph,
                               max_iter=args.max_iter, restraint_k=args.restraint_k,
                               restrain_sidechains=not args.relax_sidechains,
                               hetatm_heavy=hetatm_heavy_for_walls,
                               has_hydrogens=True)
            if ok:
                _ok(f"Minimization done  →  {minimized_pdb.name}")
                if args.amber_his:
                    # Re-normalise HIS → HID/HIE/HIP for AMBER-compatible output.
                    # OpenMM's PDBFile.writeFile writes HIS for all variants;
                    # re-normalise based on H-atom positions to restore the
                    # correct HID/HIE/HIP name.
                    _min_norm = tmp / 'minimized_his_norm.pdb'
                    n_min_norm = step_normalize_his(minimized_pdb, _min_norm)
                    shutil.copy(_min_norm, minimized_pdb)
                    if n_min_norm:
                        _info(f"HIS names re-normalised in minimized structure: {n_min_norm}")
                if args.clash_check:
                    n_after = _count_clashes(minimized_pdb)
                    stats['clashes_after'] = n_after
                    if n_after >= 0:
                        _info(f"Clashes after minimization: {n_after}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    _print_summary(stats, args, prepared_pdb,
                   minimized_pdb if args.minimize else None, elapsed)

    # ── Write log ─────────────────────────────────────────────────────────────
    log_path.write_text('\n'.join(_log_lines) + '\n', encoding='utf-8')
    _info(f"Log written  →  {log_path}")


if __name__ == '__main__':
    main()
