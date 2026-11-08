"""A fixed parent, its possible ion sites, and explicit chemical provenance.

An empty mobile site remains a site.  The parent therefore contains every
candidate ion and every fixed host atom; a generated occupancy only controls
which candidate ions are present in a deletion structure.  Geometry hashes are
independent of NPZ compression and never substitute for a teacher/protocol ID.
"""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import canonical_mask, target_indices, validate_permutation_group


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass
class ParentSpec:
    """Portable fixed-parent schema; Cartesian positions/cell have units Å.

    ``mobile_indices`` indexes the full parent, whereas permutations and optional
    site groupings use local mobile-site indices.  PBC is explicit.  A metadata
    protocol may bind teacher/checkpoint/head/precision and geometry treatment;
    it is deliberately not inferred from composition.
    """
    numbers: np.ndarray
    positions_A: np.ndarray
    cell_A: np.ndarray
    mobile_indices: np.ndarray
    permutations: np.ndarray
    pbc: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=bool))
    site_groups: np.ndarray | None = None
    source_sha256: str | None = None
    symmetry_audit: dict = field(default_factory=dict)
    protocol: dict = field(default_factory=dict)

    def __post_init__(self):
        self.numbers = np.asarray(self.numbers, dtype=np.int64)
        self.positions_A = np.asarray(self.positions_A, dtype=float)
        self.cell_A = np.asarray(self.cell_A, dtype=float)
        self.mobile_indices = np.asarray(self.mobile_indices, dtype=np.int64)
        self.permutations = np.asarray(self.permutations, dtype=np.int64)
        self.pbc = np.asarray(self.pbc, dtype=bool)
        if self.numbers.ndim != 1 or len(self.numbers) == 0:
            raise ValueError("numbers must identify every full-parent atom")
        if self.positions_A.shape != (len(self.numbers), 3) or self.cell_A.shape != (3, 3):
            raise ValueError("parent positions/cell shapes are inconsistent")
        if not np.isfinite(self.positions_A).all() or not np.isfinite(self.cell_A).all():
            raise ValueError("parent geometry is nonfinite")
        if abs(np.linalg.det(self.cell_A)) < 1e-12 or self.pbc.shape != (3,):
            raise ValueError("nondegenerate cell and three PBC flags are required")
        if np.any((self.numbers < 1) | (self.numbers > 118)):
            raise ValueError("parent must contain physical atomic numbers, not padding atoms")
        if (self.mobile_indices.ndim != 1 or not len(self.mobile_indices)
                or len(np.unique(self.mobile_indices)) != len(self.mobile_indices)
                or np.any((self.mobile_indices < 0) | (self.mobile_indices >= len(self.numbers)))):
            raise ValueError("mobile_indices must be unique full-parent indices")
        if self.permutations.ndim != 2 or self.permutations.shape[1] != self.n_sites:
            raise ValueError("one local mobile-site permutation per operation is required")
        validate_permutation_group(self.permutations)
        if self.site_groups is not None:
            self.site_groups = np.asarray(self.site_groups, dtype=np.int64)
            if self.site_groups.shape != (self.n_sites,):
                raise ValueError("site_groups must label every mobile site")
            relation = self.site_groups[:, None] == self.site_groups[None, :]
            for operation in self.permutations:
                mapped = self.site_groups[operation]
                if not np.array_equal(relation, mapped[:, None] == mapped[None, :]):
                    raise ValueError("site grouping is not preserved as a partition by retained parent symmetries")

    @property
    def n_sites(self):
        return len(self.mobile_indices)

    @property
    def host_indices(self):
        return np.setdiff1d(np.arange(len(self.numbers)), self.mobile_indices)

    def _geometry_payload(self):
        return dict(schema="cathode-parent-v1", numbers=self.numbers.tolist(),
                    positions_A=self.positions_A.tolist(), cell_A=self.cell_A.tolist(),
                    pbc=self.pbc.tolist(), mobile_indices=self.mobile_indices.tolist(),
                    occupancy_convention="one=present; zero=vacant candidate site",
                    permutations=self.permutations.tolist(),
                    site_groups=None if self.site_groups is None else self.site_groups.tolist())

    @property
    def parent_hash(self):
        return _json_hash(self._geometry_payload())

    def to_dict(self):
        return dict(self._geometry_payload(), parent_hash=self.parent_hash,
                    source_sha256=self.source_sha256, symmetry_audit=self.symmetry_audit,
                    protocol=self.protocol,
                    scope="Fixed geometry/site definition; not a claim of chemistry or size transfer")

    @classmethod
    def from_dict(cls, value):
        if value.get("schema") != "cathode-parent-v1":
            raise ValueError("unrecognized ParentSpec schema")
        result = cls(**{k: value[k] for k in ("numbers", "positions_A", "cell_A", "mobile_indices", "permutations", "pbc")},
                     site_groups=value.get("site_groups"), source_sha256=value.get("source_sha256"),
                     symmetry_audit=value.get("symmetry_audit", {}), protocol=value.get("protocol", {}))
        if value.get("parent_hash") != result.parent_hash:
            raise ValueError("ParentSpec geometry hash mismatch")
        return result

    def write(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")

    @classmethod
    def from_legacy(cls, parent, source_sha256=None, *, validate_symmetry=True, symprec_A=1e-5):
        """Adapt stored parents without copying large mask tables.

        Legacy parents omit PBC; their original preparation required 3-D
        periodicity, so that assumption is explicitly recorded. ``symprec_A``
        is a provisional coordinate-matching audit tolerance in Å, not an energy
        degeneracy criterion.  All proposed mobile permutations must be induced
        by a symmetry of the *full chemically labelled parent*.
        """
        idx = target_indices(parent)
        result = cls(parent["numbers"], parent["positions_A"], parent["cell_A"], idx,
                     parent.get("permutations", np.arange(len(idx))[None]),
                     parent.get("pbc", np.ones(3, bool)),
                     site_groups=parent.get("site_groups"), source_sha256=source_sha256)
        if validate_symmetry:
            result.symmetry_audit = result.audit_host_symmetry(symprec_A)
        else:
            result.symmetry_audit = dict(status="NOT_CHECKED", reason="Explicit caller opt-out; do not claim chemical symmetry validation")
        result.symmetry_audit["pbc_provenance"] = "stored" if "pbc" in parent else "legacy preparation required 3-D periodic parent"
        return result

    def audit_host_symmetry(self, tolerance_A=1e-5):
        """Match retained site actions to species-preserving space-group actions.

        For an operation accepted within this tiny coordinate tolerance, wrapped
        fractional differences are near zero, so the cell-metric residual can be
        checked directly. Bijection and chemical identity are verified explicitly;
        approximate target-only distance preservation would not prove host symmetry.
        """
        import spglib
        if tolerance_A <= 0:
            raise ValueError("symmetry tolerance must be positive Å")
        identity = np.arange(self.n_sites)
        if not np.all(self.pbc):
            if len(self.permutations) != 1 or not np.array_equal(self.permutations[0], identity):
                raise ValueError("non-3-D-periodic ParentSpec currently supports audited identity symmetry only")
            return dict(status="PASS", operations=1, maximum_match_residual_A=0., tolerance_A=tolerance_A,
                        method="identity operation of the full host", pbc=self.pbc.tolist())
        frac = np.linalg.solve(self.cell_A.T, self.positions_A.T).T % 1.
        ops = spglib.get_symmetry((self.cell_A, frac, self.numbers), symprec=tolerance_A)
        if ops is None:
            raise ValueError("no full chemically labelled parent symmetry found")
        allowed = {}
        local = {int(j): i for i, j in enumerate(self.mobile_indices)}
        for rot, trans in zip(ops["rotations"], ops["translations"]):
            mapped = frac @ rot.T + trans
            full = np.empty(len(self.numbers), dtype=int)
            max_residual = 0.
            valid = True
            for z in np.unique(self.numbers):
                group = np.flatnonzero(self.numbers == z)
                delta = mapped[group, None, :] - frac[None, group, :]
                delta -= np.round(delta)
                distance = np.linalg.norm(delta @ self.cell_A, axis=-1)
                matches = distance.argmin(axis=1)
                if len(np.unique(matches)) != len(group):
                    valid = False
                    break
                max_residual = max(max_residual, float(distance[np.arange(len(group)), matches].max()))
                full[group] = group[matches]
            if not valid or max_residual > tolerance_A:
                continue
            if not all(int(full[i]) in local for i in self.mobile_indices):
                continue
            action = tuple(local[int(full[i])] for i in self.mobile_indices)
            if action not in allowed or max_residual < allowed[action]:
                allowed[action] = max_residual
        missing = [p.tolist() for p in self.permutations if tuple(p) not in allowed]
        if missing:
            raise ValueError(f"{len(missing)} mobile operations are not verified symmetries of the full host chemistry")
        return dict(status="PASS", operations=len(self.permutations), full_space_group_operations=len(ops["rotations"]),
                    maximum_match_residual_A=max(allowed[tuple(p)] for p in self.permutations),
                    tolerance_A=tolerance_A, tolerance_provenance="Provisional numerical coordinate-match audit; not energetic or magnetic equivalence",
                    method="spglib operations plus explicit species-preserving full-parent bijection",
                    spglib_version=spglib.__version__)

    def canonical_key(self, occupancy):
        """String-based canonicalization has no fixed-width site-count limit."""
        return canonical_mask(occupancy, self.permutations)[0]

    def example_record(self, occupancy, *, teacher, checkpoint, head, precision,
                       geometry_protocol, condition_schema_id, descriptor_schema_id=None):
        """Bind one proposed mask to an explicit energy/geometry label identity.

        The record contains no fabricated physical label. A subsequent oracle
        result must retain these fields and the exact evaluated coordinates.
        Counts and host size follow from this parent's indices, never from Li/48
        assumptions. Multiple mobile species need additional per-species count
        constraints before claiming composition-controlled generation.
        """
        occupancy = np.asarray(occupancy)
        if occupancy.shape != (self.n_sites,) or not np.isin(occupancy, [0, 1]).all():
            raise ValueError("one binary present-ion value per candidate site is required")
        fields = dict(teacher=teacher, checkpoint=checkpoint, head=head, precision=precision,
                      geometry_protocol=geometry_protocol, condition_schema_id=condition_schema_id)
        if any(not isinstance(v, str) or not v for v in fields.values()):
            raise ValueError("teacher/checkpoint/head/precision/protocol/schema must be explicit nonempty strings")
        return dict(schema="fixed-parent-occupation-example-v1", parent_hash=self.parent_hash,
                    target_count=int(occupancy.sum()), candidate_sites=self.n_sites,
                    present_atoms=len(self.host_indices) + int(occupancy.sum()),
                    occupancy=occupancy.astype(int).tolist(), canonical_vacancy_key=self.canonical_key(occupancy),
                    descriptor_schema_id=descriptor_schema_id, **fields)

