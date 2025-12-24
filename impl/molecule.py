from abc import ABCMeta, abstractmethod
from enum import Enum
import sys
import dataclasses
import numpy as np

@dataclasses.dataclass(frozen=True)
class LocalEnergyConstant:
    E_aa: float
    E_ww: float
    E_aw: float
    E_asc: float
    E_ash: float
    E_asn: float
    E_wsc: float
    E_wsh: float
    E_wsn: float
    E_hpi: float

class MoleculeKind(Enum):
    SoapKind = 1
    WaterKind = 2
    AirKind = 3

class Molecule(metaclass=ABCMeta):
    @abstractmethod
    def encode(self):
        pass
    @abstractmethod
    def calc_self_energy(self, neighbor):
        pass

class Soap:
    def __init__(self, rng=None, dir=None):
        if rng:
            self.dir = rng.choice(7, size=1)
        elif dir:
            self.dir = dir
        else:
            print("cannot assign Soap dir.")
            sys.exit()

    def encode(self):
        return [MoleculeKind.SoapKind, self.dir]

    def calc_self_energy(self, neighbor):
        pass

class Water:
    def encode(self):
        return [MoleculeKind.WaterKind, -1]

    def calc_self_energy(self, neighbor):
        pass

class Air:
    def encode(self):
        return [MoleculeKind.AirKind, -1]

    def calc_self_energy(self, neighbor):
        pass

Molecule.register(Soap)
Molecule.register(Water)
Molecule.register(Air)

class MCMCUtl:
    def decode(self, encoded):
        match encoded[0]:
            case MoleculeKind.SoapKind:
                return Soap(dir=encoded[1])
            case MoleculeKind.WaterKind:
                return Water()
            case MoleculeKind.AirKind:
                return Air()
            case _:
                print("invalid decode.")
                sys.exit()

    def calc_neighbor_energy(self, neighbor):
        energy = 0.0
        for row_idx in range(1, 4):
            for col_idx in range(1, 4):
                energy = energy + self.decode(neighbor[row_idx, col_idx]).calc_self_energy(neighbor[row_idx-1:row_idx+1+1, col_idx-1:col_idx+1+1])
        return energy

    def MCMC_step(self, original_energy, may_swap_energy, temp_scale, rng):
        zero_to_one = rng.random()
        return zero_to_one < np.exp(-((original_energy-may_swap_energy)/temp_scale))
        

    def try_local_swap(self, neighbor, temp_scale, rng):
        may_swap_indicator = rng.choice(7, size=1)
        d_idx = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]][may_swap_indicator]
        may_swap_idx = [2+d_idx[0], 2+d_idx[1]]
        original_encoded, may_swap_encoded = neighbor[2, 2], neighbor[may_swap_idx[0], may_swap_idx[1]]
        may_swap_neighbor = neighbor.copy()
        may_swap_neighbor[2, 2] = may_swap_encoded
        may_swap_neighbor[may_swap_idx[0], may_swap_idx[1]] = original_encoded
        original_energy = self.calc_neighbor_energy(neighbor)
        may_swap_energy = self.calc_neighbor_energy(may_swap_neighbor)
        is_swap = self.MCMC_step(original_energy, may_swap_energy, temp_scale, rng)
        if is_swap:
            return may_swap_neighbor
        else:
            return neighbor

