"""Generate fixed-count occupancy attempts from a trusted frozen checkpoint.

Every attempt is saved, including duplicate or previously known symmetry orbits.
A low-energy class is a requested condition, not a measured output energy.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .models import make_model
from .runtime import device_name, digest, trusted_checkpoint
from ..runtime import require_compute_allocation


def canonical_masks(occupancy, permutations):
    """Return lexicographically least vacancy strings and orientation witnesses.

    Parent permutations map source site to destination; indexing by their inverse
    applies each operation. No force or coordinate vector is rotated here.
    """
    x = np.asarray(occupancy)
    perms = np.asarray(permutations)
    n = x.shape[1]
    if perms.ndim != 2 or perms.shape[1] != n or not np.array_equal(
            np.sort(perms, axis=1), np.tile(np.arange(n), (len(perms), 1))):
        raise ValueError('Parent permutations must be valid site permutations')
    inv = np.argsort(perms, axis=1)
    keys, witnesses = [], []
    for row in x:
        candidates = (1 - row)[inv]
        strings = [''.join(map(str, candidate.astype(int))) for candidate in candidates]
        witness = min(range(len(strings)), key=strings.__getitem__)
        keys.append(strings[witness])
        witnesses.append(witness)
    return np.asarray(keys), np.asarray(witnesses, dtype=np.int32)


def load_model(checkpoint, device):
    ck = trusted_checkpoint(checkpoint, device)
    cfg = ck.get('configuration')
    if cfg is None:
        cfg = json.loads(Path(checkpoint).with_name('configuration.json').read_text())
    kind = cfg.get('kind', 'dense_count')
    # Original dense trainer used kind=cvae; its state_dict names are unchanged.
    if kind in {'cvae', 'composition', 'composition_only'}:
        kind = 'dense_bce'
    model = make_model(kind, n_sites=cfg.get('n_sites', 48),
                       condition_dim=cfg.get('condition_dim', 2),
                       latent_dim=cfg.get('latent_dim', 8),
                       hidden_dim=cfg.get('hidden_dim', 64)).to(device)
    model.load_state_dict(ck['state_dict'], strict=True)
    model.eval()
    return model, cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--n-li', '--n-target', dest='n_li', type=int, required=True)
    p.add_argument('--count', type=int, default=512)
    p.add_argument('--condition', type=int, choices=[0, 1], default=1)
    p.add_argument('--sampling', choices=['auto', 'exact', 'topk'], default='auto')
    p.add_argument('--seed', type=int, default=20261106)
    p.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    require_compute_allocation()
    a.device = device_name(a.device)
    if a.output.suffix != '.npz':
        raise ValueError('--output must end in .npz')
    if a.count < 1:
        raise ValueError('--count must be positive')
    unique_output=a.output.with_name(a.output.stem+'.unique.npz')
    for output in [a.output, unique_output, a.output.with_suffix('.attempts.csv'), a.output.with_suffix('.json')]:
        if output.exists():
            raise FileExistsError(output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, cfg = load_model(a.checkpoint, a.device)
    if cfg.get("parent_sha256") and cfg["parent_sha256"] != digest(a.parent):
        raise ValueError("Parent hash differs from checkpoint parent; train a new model for another material/cell")
    if not 0 <= a.n_li <= model.n_sites:
        raise ValueError('Target count is outside checkpoint parent size')
    with np.load(a.parent, allow_pickle=False) as z:
        perms = z['permutations']
    if perms.shape[1] != model.n_sites:
        raise ValueError('Checkpoint and parent site dimensions differ')
    sampling = a.sampling
    if sampling == 'auto':
        sampling = 'topk' if str(cfg.get('projection', '')).startswith('top-') else 'exact'
    c = torch.tensor([[a.n_li / model.n_sites, a.condition]], dtype=torch.float32,
                     device=a.device).repeat(a.count, 1)[:, :model.condition_dim]
    rng = torch.Generator(device=a.device).manual_seed(a.seed)
    with torch.no_grad():
        if sampling == 'topk':
            if not cfg.get('kind', '').startswith(('dense', 'cvae', 'composition')):
                raise ValueError('topk is only defined for the independent dense decoder')
            z = torch.randn((a.count, model.latent_dim), generator=rng, device=a.device)
            logits = model.decode_logits(z, c)
            logits_np = logits.cpu().numpy()
            np_rng = np.random.default_rng(a.seed)
            order = np.lexsort((np_rng.random(logits_np.shape), -logits_np), axis=1)
            occupancy = np.zeros(logits_np.shape, dtype=np.uint8)
            np.put_along_axis(occupancy, order[:, :a.n_li], 1, axis=1)
        else:
            occupancy = model.sample(c, a.n_li, generator=rng).cpu().numpy().astype(np.uint8)
    assert np.all(occupancy.sum(1) == a.n_li)
    keys, witness = canonical_masks(occupancy, perms)
    canonical_occupancy = 1 - np.asarray([[int(bit) for bit in key] for key in keys], np.uint8)
    width = (model.n_sites + 3) // 4
    ids = np.asarray([f'mask{model.n_sites}_{int(k, 2):0{width}x}' for k in keys])
    np.savez_compressed(a.output, occupancy=occupancy, canonical_occupancy=canonical_occupancy,
                        canonical_vacancy_mask=keys, masks=1-occupancy,
                        canonicalizing_inverse_permutation_index=witness,
                        structure_ids=ids, n_Li=np.full(a.count, a.n_li),
                        n_target=np.full(a.count,a.n_li), seed=a.seed)
    unique_keys, first=np.unique(keys,return_index=True)
    unique_occupancy=canonical_occupancy[first]
    _, unique_witness=canonical_masks(unique_occupancy,perms)
    np.savez_compressed(unique_output,occupancy=unique_occupancy,canonical_occupancy=unique_occupancy,
                        canonical_vacancy_mask=unique_keys,masks=1-unique_occupancy,
                        canonicalizing_inverse_permutation_index=unique_witness,structure_ids=ids[first],
                        n_Li=np.full(len(first),a.n_li),n_target=np.full(len(first),a.n_li),
                        orientation=np.asarray('canonical'),parent_sha256=np.asarray(digest(a.parent)),seed=a.seed)
    with a.output.with_suffix('.attempts.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['method','seed','n_Li','attempt','scoring_structure_id','canonical_mask'])
        w.writeheader()
        for i, (sid, key) in enumerate(zip(ids, keys)):
            w.writerow(dict(method=cfg.get('kind', 'legacy_cvae'), seed=a.seed, n_Li=a.n_li,
                            attempt=i, scoring_structure_id=sid, canonical_mask=key))
    receipt = dict(attempts=a.count, unique_orbits=len(set(keys)), n_Li=a.n_li,
                   sampling=sampling, seed=a.seed, device=a.device,
                   checkpoint_sha256=digest(a.checkpoint), parent_sha256=digest(a.parent),
                   parent_identity_verified=bool(cfg.get("parent_sha256")),
                   parent_identity_note="Hash-verified" if cfg.get("parent_sha256") else "Historical checkpoint lacks parent hash; supplied parent dimensions verified, chemical identity not independently certified",
                   output_sha256=digest(a.output), unique_output=str(unique_output),
                   unique_output_sha256=digest(unique_output), model_configuration=cfg,
                   scope='Prior occupancy proposals; no geometry optimization or measured energy; duplicates retained')
    a.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ['attempts','unique_orbits','n_Li','sampling']}, indent=2))


if __name__ == '__main__':
    main()
