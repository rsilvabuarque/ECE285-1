"""Frozen teachers used in the November 2026 comparison, not moving defaults.

SHA256 values come from the completed campaign's checkpoint receipt. They make
download corruption and an upstream checkpoint replacement visible before use.
Model licenses remain those of their respective authors.
"""

MODELS = {
    "uma": {
        "name": "UMA-S-1.2.1", "package": "fairchem-core", "version": "2.25.0",
        "backend": "uma", "head": "omat", "repo": "facebook/UMA",
        "revision": "ce3b2744ff57c0f49a010b763efcbfe9ba82e2e1",
        "files": {"checkpoints/uma-s-1p2p1.pt": "b2673b85037b075674c25f55c34ffe1ff1e15db924be977b10a184765df0d5ce"},
        "entry": "checkpoints/uma-s-1p2p1.pt",
        "docs": "https://facebookresearch.github.io/fairchem/ase-calculator/",
    },
    "mace": {
        "name": "MACE-MH-1", "package": "mace-torch", "version": "0.3.16",
        "backend": "mace", "head": "omat_pbe",
        "url": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model",
        "files": {"mace-mh-1.model": "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47"},
        "entry": "mace-mh-1.model",
        "docs": "https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html",
    },
    "sevennet": {
        "name": "SevenNet-Omni", "package": "sevenn", "version": "0.13.0",
        "backend": "sevennet", "head": "mpa",
        "url": "https://github.com/MDIL-SNU/SevenNet/releases/download/v0.12.0.cp/checkpoint_sevennet_omni.pth",
        "files": {"checkpoint_sevennet_omni.pth": "ca81bd3aac9fc2696c93dd386615f5a0fe41b92ab9ed7f69fa9526baaa9bab64"},
        "entry": "checkpoint_sevennet_omni.pth",
        "docs": "https://github.com/MDIL-SNU/SevenNet",
    },
    "m3gnet": {
        "name": "M3GNet-PES-MatPES-PBE-2025.2", "package": "matgl", "version": "4.1.4",
        "backend": "matgl", "head": "MatPES-PBE", "repo": "materialyze/M3GNet-PES-MatPES-PBE-2025.2",
        "revision": "a972e0dd982eaf2a4cf1e5c8de2c9e78c0013f72", "entry": ".",
        "files": {
            "model.json": "96266a10e1b7f15baaed9cb6329c28138d2d3dbec09d754f60c42bd0812359e1",
            "model.pt": "8204c34dc4df513e03d080017c30d7628fa3e4fcc13b9fe47076da0bcd3129da",
            "state.pt": "ba5b96249e366456cb3e518fcc93f58c49353f564c79648f309b916926b10fa0",
        },
        "docs": "https://github.com/materialyzeai/matgl",
    },
    "chgnet": {
        "name": "CHGNet-PES-MatPES-PBE-1M-2026.9", "package": "matgl", "version": "4.1.4",
        "backend": "matgl", "head": "MatPES-PBE", "repo": "materialyze/CHGNet-PES-MatPES-PBE-1M-2026.9",
        "revision": "b38b2e9523479a4601e46e9ae5362845f055111e", "entry": ".",
        "files": {
            "model.json": "ef0a447a9989168ee326605fb21a4e9c9807ee15a61f6682f7be474ba028dc9e",
            "model.pt": "523d3da637687e82e2a4645197810e7aaaf89c51421d4d1708cfcd98f74370a6",
            "state.pt": "e9da73c92dc50734db276029936768e27a26a506fc226c98de2f08106488c6d4",
        },
        "docs": "https://github.com/materialyzeai/matgl",
    },
}
