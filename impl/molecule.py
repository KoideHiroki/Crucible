from abc import ABCMeta, abstractmethod
from enum import Enum
import sys
import dataclasses
import numpy as np

dirctions = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
dirctions_normal = [np.asarray(d)/np.linalg.norm(d) for d in dirctions]
def find_dir(pos):
    ret = pos.copy()
    ret[0] = ret[0] - 1
    ret[1] = ret[1] - 1
    return dirctions.index(ret)

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
    E_sspa: float
    E_ssta: float
    E_sshi: float
    E_ssn: float

class InteractionHelpers:
    def __init__(self):
        self.lec = LocalEnergyConstant()
    def as_interaction_energy(self, soap, pos):
        if self.is_xsc_interaction(soap, pos):
            return self.lec.E_asc
        elif self.is_xsh_interaction(soap, pos):
            return self.lec.E_ash
        elif self.is_xsn_interaction(soap, pos):
            return self.lec.E_asn
        else:
            print("invalid Air-Soap interaction.")
            sys.exit()

    def ws_interaction_energy(self, soap, pos):
        if self.is_xsc_interaction(soap, pos):
            return self.lec.E_wsc
        elif self.is_xsh_interaction(soap, pos):
            return self.lec.E_wsh
        elif self.is_xsn_interaction(soap, pos):
            return self.lec.E_wsn
        else:
            print("invalid Water-Soap interaction.")
            sys.exit()

    def ss_interaction_energy(self, soap, other_soap, pos):
        if self.is_sspa_interaction(soap, other_soap, pos):
            return self.lec.E_sspa
        elif self.is_ssta_interaction(soap, other_soap, pos):
            return self.lec.E_ssta
        elif self.is_sshi_interaction(soap, other_soap, pos):
            return self.lec.E_sshi
        else:
            return self.lec.E_ssn

    def is_xsc_interaction(self, sope, pos):
        pass

    def is_xsh_interaction(self, soap, pos):
        pass

    def is_xsn_interaction(self, soap, pos):
        pass

    def is_sspa_interaction(self, soap, other_sope, pos):
        pass

    def is_ssta_interaction(self, soap, other_sope, pos):
        pass

    def is_sshi_interaction(self, soap, other_sope, pos):
        pass

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
            self.dir = rng.choice(8)
        elif dir:
            self.dir = dir
        else:
            print("cannot assign Soap dir.")
            sys.exit()

    def encode(self):
        return [MoleculeKind.SoapKind, self.dir]

    def calc_self_energy(self, neighbor):
        energy = 0.0
        mcmc_utl = MCMCUtl()
        interaction_helpers = InteractionHelpers()
        for row_idx in range(0, 3):
            for col_idx in range(0, 3):
                if row_idx == 1 and col_idx == 1:
                    continue
                ngb = neighbor[row_idx, col_idx]
                pos = find_dir([row_idx, col_idx])
                match ngb[0]:
                    case MoleculeKind.WaterKind:
                        energy = energy + interaction_helpers.ws_interaction_energy(self, (pos+4)%8)
                    case MoleculeKind.AirKind:
                        energy = energy + interaction_helpers.as_interaction_energy(self, (pos+4)%8)
                    case MoleculeKind.SoapKind:
                        other_soap = mcmc_utl.decode(ngb)
                        energy = energy + interaction_helpers.ss_interaction_energy(self, other_soap, pos)
        return energy

class Water:
    def encode(self):
        return [MoleculeKind.WaterKind, -1]

    def calc_self_energy(self, neighbor):
        energy = 0.0

        for row_idx in range(0, 3):
            for col_idx in range(0, 3):
                if row_idx == 1 and col_idx == 1:
                    continue
                ngb = neighbor[row_idx, col_idx]
                match ngb[0]:
                    case MoleculeKind.WaterKind:
                        energy = energy + self.lec.E_ww
                    case MoleculeKind.AirKind:
                        energy = energy + self.lec.E_aw
                    case MoleculeKind.SoapKind:
                        mcmc_utl = MCMCUtl()
                        soap = mcmc_utl.decode(ngb)
                        interaction_helpers = InteractionHelpers()
                        pos = find_dir([row_idx, col_idx])
                        energy = energy + interaction_helpers.ws_interaction_energy(soap, pos)
        return energy

class Air:
    def encode(self):
        return [MoleculeKind.AirKind, -1]

    def calc_self_energy(self, neighbor):
        energy = 0.0
        for row_idx in range(0, 3):
            for col_idx in range(0, 3):
                if row_idx == 1 and col_idx == 1:
                    continue
                ngb = neighbor[row_idx, col_idx]
                match ngb[0]:
                    case MoleculeKind.WaterKind:
                        energy = energy + self.lec.E_ww
                    case MoleculeKind.AirKind:
                        energy = energy + self.lec.E_aw
                    case MoleculeKind.SoapKind:
                        mcmc_utl = MCMCUtl()
                        soap = mcmc_utl.decode(ngb)
                        interaction_helpers = InteractionHelpers()
                        pos = find_dir([row_idx, col_idx])
                        energy = energy + interaction_helpers.as_interaction_energy(soap, pos)
        return energy

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
        may_swap_indicator = rng.choice(8)
        d_idx = dirctions[may_swap_indicator]
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

