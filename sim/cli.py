"""Command-line entry: python -m sim.cli run --config configs/default.yaml --ticks 1000"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import SimConfig
from .model import OralBiofilmModel
from .species import CATALOG


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--config", type=str, default=None)
    p_run.add_argument("--ticks", type=int, default=None)
    p_run.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    if args.cmd == "run":
        cfg = SimConfig.from_yaml(args.config) if args.config else SimConfig()
        if args.ticks is not None:
            cfg.ticks = args.ticks
        out_dir = Path(args.out or cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        model = OralBiofilmModel(cfg)
        traj = model.run()
        arrays = traj.to_arrays()

        # HDF5 dump
        h5_path = out_dir / "trajectory.h5"
        with h5py.File(h5_path, "w") as f:
            for k, v in arrays.items():
                f.create_dataset(k, data=v)
            f.attrs["n_species"] = len(CATALOG)
            f.attrs["species_ids"] = [s.id for s in CATALOG]

        # Quick plot — total biomass per species over time
        comp = arrays["composition"]   # (T, N_SPECIES, N_DEPTHS)
        total_per_species = comp.sum(axis=2)
        t_axis = arrays["tick"]
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, s in enumerate(CATALOG):
            ax.plot(t_axis, total_per_species[:, i], label=s.name, alpha=0.8, linewidth=1)
        ax.set_xlabel("tick")
        ax.set_ylabel("total biomass")
        ax.set_title("Species succession over time")
        ax.legend(loc="upper left", fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "succession.png", dpi=120)

        # Coherence index + dysbiosis index
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t_axis, arrays["NO"], label="NO", color="#2c7a4b")
        axes[0].plot(t_axis, arrays["H2S"], label="H2S", color="#8a2c2c")
        axes[0].set_ylabel("coherence axis (total field)")
        axes[0].legend()
        axes[1].plot(t_axis, arrays["dysbiosis"], color="#c84a4a")
        axes[1].set_ylabel("dysbiosis index (red+orange / total)")
        axes[1].set_xlabel("tick")
        fig.tight_layout()
        fig.savefig(out_dir / "coherence_and_dysbiosis.png", dpi=120)

        # Bridge vs keystone betweenness — the bridge-asymmetry observable
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_axis, arrays["Fn_betweenness"], label="F. nucleatum (bridge)", color="#e08854")
        ax.plot(t_axis, arrays["Pg_betweenness"], label="P. gingivalis (keystone)", color="#c84a4a")
        ax.set_xlabel("tick")
        ax.set_ylabel("realized betweenness")
        ax.set_title("Bridge vs keystone — the Bridge-Asymmetry observable")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "bridge_vs_keystone.png", dpi=120)

        print(f"Wrote trajectory to {h5_path}")
        print(f"Wrote plots to {out_dir}/")
        print(f"Final tick: {model.tick}  |  Live agents: {len(model.agents_alive)}")


if __name__ == "__main__":
    main()

