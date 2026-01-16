#!/usr/bin/env python3
"""
Generates a series of initial conditions for shock tube tests.

```
python ics.py -h
```

To see 
"""
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(
                    prog='ics.py',
                    description="""Generates the initial conditions for RICH
                    test problem ShockTube. Creates the corresponding folders
                    and initial condition files for RICH.""",
                    epilog='')
parser.add_argument('output_dir', type=str, help='Output directory to create new folders and files.')
parser.add_argument('-n', '--number', type=int, help='Number of initial conditions it will generate. Default to 10.', 
                    default=10)
args = parser.parse_args()

output_dir = args.output_dir
N = args.number


def P_hugoniot(rho, P_ref, rho_ref, gamma=5/3): # uses rho2, P2 as reference point
    return P_ref * ((gamma + 1)*rho - (gamma - 1)*rho_ref) / ((gamma + 1)*rho_ref - (gamma - 1)*rho)

def write_number(file, number, overwrite=False):
    if overwrite is False and os.path.isfile(file):
        with open(file, 'r') as f:
            old_number = float(f.read())
    else:
        old_number = -1

    if number != old_number:
        with open(file, 'w') as f:
            f.write(str(number))


if __name__ == '__main__':
    R = np.linspace(1+1e-3, 4-1e-5, N, endpoint=True) # compression ratio
    DL = np.ones_like(R)          # left density
    DR = DL / R
    PL = np.ones_like(R)
    PR = P_hugoniot(DR, P_ref=PL, rho_ref=DL)

    for i in range(N):
        dir = f'PL{PL[i]:.1f}PR{PR[i]:.3f}DL{DL[i]:.1f}DR{DR[i]:.3f}'
        dir = os.path.join(output_dir, dir)
        os.makedirs(dir, exist_ok=True)

        for file, param in zip(
            ['leftpressure.txt', 'rightpressure.txt', 
            'leftdensity.txt', 'rightdensity.txt'],
            [PL[i], PR[i], DL[i], DR[i]],
            ):
            file = os.path.join(dir, file)
            write_number(file, param, overwrite=False)

    print('Done!')
