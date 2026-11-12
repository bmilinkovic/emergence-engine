"""Cross-check repaired temporal DD with a high-resolution spectral integral."""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from multiscale import canonical_band_value, prepare_var
from multiscale.model import orthonormalise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("models", type=Path)
    parser.add_argument("temporal", type=Path)
    parser.add_argument("--fres", type=int, default=4096)
    args = parser.parse_args()
    errors = []
    for mat_file in sorted(args.reference.glob("*/*_ssdi_spectral_domain_canonical.mat")):
        condition = mat_file.parent.name
        subject = mat_file.name.removesuffix("_ssdi_spectral_domain_canonical.mat")
        with np.load(args.models / condition / f"{subject}.npz") as model:
            a, v = model["a"], model["v"]
        archived = loadmat(mat_file, simplify_cells=True)["canonical"]
        with np.load(args.temporal / condition / f"{subject}.npz") as repaired:
            _, _, _, _, h, freqs = prepare_var(a, v, int(archived["meta"]["fs"]),
                                                args.fres)
            root = np.linalg.cholesky(v)
            indices = np.arange(len(freqs))
            for m in range(1, a.shape[0]):
                for band in ("delta", "theta", "alpha", "beta", "gamma"):
                    bases = np.asarray(archived[f"{band}_Lopto_m{m}"])
                    basis = bases[:, 0, None] if m == 1 else bases[:, :, 0]
                    l = orthonormalise(root.T @ basis)
                    spectral = canonical_band_value(l, h, freqs, indices)
                    temporal = repaired[f"{band}_m{m}"][0]
                    errors.append(abs(temporal-spectral))
    print(f"{len(errors)} high-resolution temporal/spectral checks; "
          f"max absolute error {max(errors):.12g}; "
          f"median {np.median(errors):.12g}")


if __name__ == "__main__":
    main()
