"""Regenerate independent MVGC2 fixtures: python script.py /path/to/MVGC2."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from scipy.io import savemat


def quote(path):
    return str(path).replace("'", "''")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mvgc2', type=Path)
    parser.add_argument('--octave', default='octave')
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures'
    rng = np.random.default_rng(481)
    a = np.array([[.8, .1], [-.05, .65]])
    c = np.array([[1, .2], [-.2, .7], [.5, -.4]])
    k = np.array([[.15, .05, .02], [-.03, .12, .04]])
    x = np.empty((3, 600, 3))
    for trial in range(3):
        z = np.zeros(2)
        for t in range(700):
            e = rng.normal(size=3)
            y = c@z+e
            z = a@z+k@e
            if t >= 100:
                x[:, t-100, trial] = y+np.array([2, -3, 1])
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        savemat(folder/'input.mat', {'X': x})
        script = f"addpath(genpath('{quote(args.mvgc2.resolve())}'));\nload('{quote(folder/'input.mat')}');\n"
        for name, horizon in [('ss_mvgc2.mat', '[4,3]'), ('ss_mvgc2_single.mat', '4')]:
            if 'single' in name:
                script += 'X = X(:,:,1);\n'
            script += f"[A,C,K,V,Z,E] = tsdata_to_ss(X,{horizon},2);\n"
            script += f"[mosvc,rmax] = tsdata_to_ssmo(X,{horizon},[]);\n"
            script += f"save('-mat7-binary','{quote(output/name)}','X','A','C','K','V','Z','E','mosvc','rmax');\n"
        (folder/'reference.m').write_text(script)
        subprocess.run([args.octave, '--quiet', '--no-gui', str(folder/'reference.m')], check=True)


if __name__ == '__main__':
    main()
