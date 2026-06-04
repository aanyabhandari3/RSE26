# RSE Beacons Architecture Extended

This project is focusing on extending the MLP structure in Beacons to Transformers handling hyperbolic PDEs. File organization is based on which repository the files go into.
Files in kann are for the kann library and have dependencies built in. Same for all other folders.

## Currently working on....
- Adding transformer.c to kann
- rollout divergence updated to work for transformer.c
- updating sweep to compare transformer.c with MLP and RNN
- adding L2 and L-inf error for MLP and RNN comparison (expand to transformer later)
- Adding linear_advection.rkt to gkylcas built for RNN. Includes code generation files.
- lit review for approximation theory for RNN and inherently transformers

## Repositories

KANN: https://github.com/attractivechaos/kann
GKYLCAS: https://github.com/ammarhakim/gkylcas/tree/main/provable-algorithms/neural_networks
