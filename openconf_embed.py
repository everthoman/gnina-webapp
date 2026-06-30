#!/usr/bin/env python3
"""Subprocess worker: takes SMILES from argv[1], writes SDF molblock to stdout."""
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from openconf import generate_conformers
from openconf.config import preset_config, ConformerConfig


def _ring_normal(positions: np.ndarray, ring_indices: list[int]) -> np.ndarray:
    """Mean unit normal of a ring."""
    pts = positions[ring_indices]
    center = pts.mean(axis=0)
    n = len(ring_indices)
    normal = np.zeros(3)
    for i in range(n):
        normal += np.cross(pts[i] - center, pts[(i + 1) % n] - center)
    norm = np.linalg.norm(normal)
    return normal / norm if norm > 1e-8 else normal


def _ring_torsion_angles(positions: np.ndarray, ring: list[int]) -> list[float]:
    """Compute all 6 consecutive torsion angles around a 6-membered ring."""
    n = len(ring)
    angles = []
    for i in range(n):
        a, b, c, d = [positions[ring[(i + j) % n]] for j in range(4)]
        b1 = b - a
        b2 = c - b
        b3 = d - c
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        norm1, norm2 = np.linalg.norm(n1), np.linalg.norm(n2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            angles.append(0.0)
            continue
        n1, n2 = n1 / norm1, n2 / norm2
        cos_t = np.clip(np.dot(n1, n2), -1.0, 1.0)
        sign = np.sign(np.dot(np.cross(n1, n2), b2))
        angles.append(float(np.degrees(np.arccos(cos_t))) * (sign if sign != 0 else 1.0))
    return angles


def _is_aromatic_ring(mol: Chem.Mol, ring: tuple[int, ...]) -> bool:
    return all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)


def _chair_fraction(mol: Chem.Mol, conf_id: int) -> float:
    """Fraction of non-aromatic 6-membered rings in true chair conformation.

    A chair has all 6 ring torsions |τ| > 30° with alternating signs.
    Boats and twist-boats fail one or both criteria.
    Aromatic rings are excluded — they are always flat, not chairs.
    Returns 1.0 when there are no non-aromatic 6-membered rings (no penalty).
    """
    positions = mol.GetConformer(conf_id).GetPositions()
    n_chair = 0
    n_rings = 0
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 6 or _is_aromatic_ring(mol, ring):
            continue
        n_rings += 1
        torsions = _ring_torsion_angles(positions, list(ring))
        if any(abs(t) < 30.0 for t in torsions):
            continue
        signs = [1 if t > 0 else -1 for t in torsions]
        alternating = all(signs[i] != signs[(i + 1) % 6] for i in range(6))
        if alternating:
            n_chair += 1
    return n_chair / n_rings if n_rings > 0 else 1.0


def _equatorial_fraction(mol: Chem.Mol, conf_id: int) -> float:
    """Fraction of heavy-atom substituents on non-aromatic 6-membered rings that are equatorial.

    Aromatic rings are excluded — their substituents are trivially in-plane and would
    dilute the signal for the aliphatic rings we care about (piperidines, cyclohexanes).
    Returns 1.0 when there are no such substituents (no penalty).
    """
    positions = mol.GetConformer(conf_id).GetPositions()
    equatorial = 0
    total = 0
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 6 or _is_aromatic_ring(mol, ring):
            continue
        normal = _ring_normal(positions, list(ring))
        ring_set = set(ring)
        for idx in ring:
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx in ring_set or nbr.GetAtomicNum() == 1:
                    continue
                bond_vec = positions[nidx] - positions[idx]
                length = np.linalg.norm(bond_vec)
                if length < 1e-6:
                    continue
                cos_angle = abs(np.dot(bond_vec / length, normal))
                total += 1
                if cos_angle < 0.7:  # axial ≈ 0.9–1.0, equatorial ≈ 0.55–0.65
                    equatorial += 1
    return equatorial / total if total > 0 else 1.0


def _flip_chair(mol: Chem.Mol, conf_id: int, ring: list[int]) -> int:
    """Add a ring-flipped conformer to mol (axial↔equatorial swap).

    Reflects each ring atom through the mean ring plane, then translates
    immediate substituents by the same delta so bond lengths are preserved.
    Returns the new conformer ID.
    """
    from rdkit.Geometry import Point3D

    positions = mol.GetConformer(conf_id).GetPositions()
    ring_set = set(ring)
    center = positions[list(ring)].mean(axis=0)
    normal = _ring_normal(positions, ring)

    new_pos = positions.copy()
    for idx in ring:
        disp = np.dot(positions[idx] - center, normal)
        new_pos[idx] = positions[idx] - 2.0 * disp * normal
    for idx in ring:
        delta = new_pos[idx] - positions[idx]
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in ring_set:
                new_pos[nidx] = positions[nidx] + delta

    new_conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        p = new_pos[i]
        new_conf.SetAtomPosition(i, Point3D(float(p[0]), float(p[1]), float(p[2])))
    return mol.AddConformer(new_conf, assignId=True)


def _weighted_equatorial_fraction(mol: Chem.Mol, conf_id: int) -> float:
    """Like _equatorial_fraction but weights N-ring exocyclic bonds 2× vs C-ring bonds.

    Piperidine/piperazine N-substituents strongly prefer equatorial in drug contexts
    and carry more pharmacophoric weight than the C4-substituent.
    """
    positions = mol.GetConformer(conf_id).GetPositions()
    weighted_eq = 0.0
    weighted_total = 0.0
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 6 or _is_aromatic_ring(mol, ring):
            continue
        normal = _ring_normal(positions, list(ring))
        ring_set = set(ring)
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            weight = 2.0 if atom.GetAtomicNum() == 7 else 1.0
            for nbr in atom.GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx in ring_set or nbr.GetAtomicNum() == 1:
                    continue
                bond_vec = positions[nidx] - positions[idx]
                length = np.linalg.norm(bond_vec)
                if length < 1e-6:
                    continue
                cos_angle = abs(np.dot(bond_vec / length, normal))
                weighted_total += weight
                if cos_angle < 0.7:
                    weighted_eq += weight
    return weighted_eq / weighted_total if weighted_total > 0 else 1.0


def _mmff_minimize(mol: Chem.Mol, conf_id: int) -> float:
    """Minimize conf_id in place with MMFF94s (UFF fallback). Returns final energy."""
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id) if props else None
    if ff is None:
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
    if ff is not None:
        ff.Minimize(maxIts=2000)
        return ff.CalcEnergy()
    return float('inf')


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: openconf_embed.py <smiles>")

    smiles = sys.argv[1]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Retry without strict sanitization (handles e.g. tetrazolate [N-] from obabel)
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except Exception:
                mol = None
    if mol is None:
        sys.exit(f"ERROR: could not parse SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    # Docking preset: wide energy window (18 kcal), 250 conformers, uniform parent
    # sampling — better basin coverage than the default config.
    base = preset_config("docking")
    cfg = ConformerConfig(**{**vars(base), 'random_seed': 0xF00D})
    ensemble = generate_conformers(mol, config=cfg)
    if ensemble.n_conformers == 0:
        sys.exit("ERROR: openconf generated 0 conformers")

    # Strategy: scan all conformers for chair geometry, then for each unique chair
    # also generate the ring-flipped conformer.  This guarantees both chair
    # conformations are evaluated even when openconf only samples one.
    # Diequatorial chairs can sit 10+ kcal above diaxial in raw energy, so we
    # cannot rely on an energy cut — we scan all conformers for chair quality.
    aliphatic_rings = [list(r) for r in mol.GetRingInfo().AtomRings()
                       if len(r) == 6 and not _is_aromatic_ring(mol, r)]

    chair_indices = [i for i in range(ensemble.n_conformers)
                     if _chair_fraction(ensemble.mol, ensemble.conf_ids[i]) > 0.9]

    if chair_indices:
        # Cap at 10 chairs to keep minimization cost bounded
        chair_indices_top = sorted(chair_indices,
                                   key=lambda i: ensemble.energies[i])[:10]
        cids_to_minimize = []
        for idx in chair_indices_top:
            conf_id = ensemble.conf_ids[idx]
            cids_to_minimize.append(conf_id)
            # Generate ring-flipped conformer(s) for each non-aromatic 6-ring
            for ring in aliphatic_rings:
                try:
                    flip_id = _flip_chair(ensemble.mol, conf_id, ring)
                    cids_to_minimize.append(flip_id)
                except Exception:
                    pass
    else:
        # No chairs: fall back to top-20 by raw energy
        top_indices = sorted(range(ensemble.n_conformers),
                             key=lambda i: ensemble.energies[i])[:20]
        cids_to_minimize = [ensemble.conf_ids[i] for i in top_indices]

    candidates = []
    for conf_id in cids_to_minimize:
        e_mmff = _mmff_minimize(ensemble.mol, conf_id)
        wef = _weighted_equatorial_fraction(ensemble.mol, conf_id)
        ef = _equatorial_fraction(ensemble.mol, conf_id)
        candidates.append((conf_id, e_mmff, wef, ef))

    # Primary: highest N-weighted equatorial fraction; secondary: highest plain
    # equatorial fraction; tertiary: lowest MMFF energy
    candidates.sort(key=lambda x: (-x[2], -x[3], x[1]))
    best_conf_id = candidates[0][0]
    print(Chem.MolToMolBlock(ensemble.mol, confId=best_conf_id))


if __name__ == "__main__":
    main()
