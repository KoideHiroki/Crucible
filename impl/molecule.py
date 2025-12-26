from abc import ABCMeta, abstractmethod
from enum import Enum
import sys
import dataclasses
import numpy as np

directions = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
def find_dir(pos):
    ret = pos.copy()
    ret[0] = ret[0] - 1
    ret[1] = ret[1] - 1
    return directions.index(ret)

@dataclasses.dataclass(frozen=True)
class LocalEnergyConstant:
    E_aa: float = 0.0
    E_ww: float = -1.0
    E_aw: float = 1.0
    E_asc: float = -0.5
    E_ash: float = 1.0
    E_asn: float = -0.2
    E_wsc: float = 1.0
    E_wsh: float = -1.2
    E_wsn: float = -0.2
    #E_sspa: float = -0.5
    #E_ssta: float = -1.5
    #E_sshi: float = -2.0
    #E_ssn: float = 0.0
    #E_sscurv: float = -2.0
    E_sspa: float = -2.5
    E_ssta: float = -3.5
    E_sshi: float = -4.0
    E_ssn: float = 0.0
    E_sscurv: float = -4.0

LEC = LocalEnergyConstant(...)

class InteractionHelpers:
    def __init__(self):
        self.lec = LEC
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
        energy = None
        if self.is_sspa_interaction(soap, other_soap, pos):
            energy = self.lec.E_sspa
        elif self.is_ssta_interaction(soap, other_soap, pos):
            energy = self.lec.E_ssta
        elif self.is_sshi_interaction(soap, other_soap, pos):
            energy = self.lec.E_sshi
        else:
            energy = self.lec.E_ssn

        if self.is_sscurv_interaction(soap, other_soap, pos):
            energy = energy + self.lec.E_sscurv
        return energy

    """
    以下、is_xxx~~~~は「その相互作用に当てはまるかどうか」でboolを返す。
        xsc
        x(AirかWater)のpos(0~7)方向にSoapがあるとき、soap.dir(0~7)がxの方向を向いているか
        xsh
        xのpos方向にSoapがあるとき、soap.dirの反対がxの方向にあるか
        xsn
        xのpos方向にSoapがあるとき、xscでもxshでもないとき
        sspa
        soapのpos方向にother_soapがあるとき、soap.dirとother_soap.dirがposの相対角度を考慮して45°違いになっているとき
        ssta
        soapのpos方向にother_soapがあるとき、soap.dirとother_soap.dirがposの相対角度を考慮して同じ向きになっているとき
        sshi
        soapのpos方向にother_soapがあるとき、sspaでもsstaでもなく、soap.dirとother_soap.dirがposの相対角度を考慮して同じ格子を指しているとき
        sscurv
        soap同士が曲線を作るか
    """
    def _ang_diff8(self, a: int, b: int) -> int:
        """8方向(0..7)の最小角差(0..4)を返す"""
        d = (a - b) % 8
        return min(d, 8 - d)

    def is_xsc_interaction(self, soap, pos):
            """
            x(Air/Water)のpos方向にSoapがあるとき、
            soap.dirがxの方向(= soap→x)を向いているか。
            ここで pos は x→soap なので、soap→x は (pos+4)%8
            """
            toward_x = (pos + 4) % 8
            return soap.dir == toward_x

    def is_xsh_interaction(self, soap, pos):
            """
            x→soap が pos のとき、soap.dir の反対(=soap.dir+4)が x 方向(soap→x)か。
            (soap.dir+4)==(pos+4) ⇔ soap.dir==pos
            """
            return soap.dir == (pos % 8)

    def is_xsn_interaction(self, soap, pos):
        return not self.is_xsc_interaction(soap, pos) and not self.is_xsh_interaction(soap, pos)

    def is_sspa_interaction(self, soap, other_soap, pos):
            """
            soapのpos方向にother_soapがあるとき、
            soap.dir と other_soap.dir が 45°違い(±1)になっているか。
            """
            return self._ang_diff8(soap.dir, other_soap.dir) == 1
            
    def is_ssta_interaction(self, soap, other_soap, pos):
            """
            soapのpos方向にother_soapがあるとき、
            soap.dir と other_soap.dir が同じ向きか。
            """
            return self._ang_diff8(soap.dir, other_soap.dir) == 0

    def is_sshi_interaction(self, soap, other_soap, pos):
        """
        soapのpos方向にother_soapがあるとき、
        2粒子を結ぶ格子軸(pos, pos+4)を
        両者が「どの角度でも」指していれば True。

        反対向き・45°・90°すべて含める。
        """
        axis = {pos % 8, (pos + 4) % 8}
        return (soap.dir in axis) and (other_soap.dir in axis)

    def is_sscurv_interaction(self, soap, other_soap, pos: int, tol: int = 1) -> bool:
        toward_other = pos % 8
        toward_self  = (pos + 4) % 8
        return (self._ang_diff8(soap.dir, toward_other) <= tol) and \
            (self._ang_diff8(other_soap.dir, toward_self) <= tol)

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
        if rng != None:
            self.dir = rng.choice(8)
        elif dir != None:
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
    def __init__(self):
        self.lec = LEC

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
    def __init__(self):
        self.lec = LEC

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
        d_idx = directions[may_swap_indicator]
        may_swap_idx = [2+d_idx[0], 2+d_idx[1]]
        original_encoded, may_swap_encoded = neighbor[2, 2], neighbor[may_swap_idx[0], may_swap_idx[1]]
        may_swap_neighbor = neighbor.copy()
        may_swap_neighbor[2, 2] = may_swap_encoded
        may_swap_neighbor[may_swap_idx[0], may_swap_idx[1]] = original_encoded
        if neighbor[2, 2][0] == MoleculeKind.SoapKind:
            diff_dir = rng.choice([-1, 0, 1])
            may_swap_neighbor[may_swap_idx[0], may_swap_idx[1]][1] = (may_swap_neighbor[may_swap_idx[0], may_swap_idx[1]][1] + diff_dir)%8
        original_energy = self.calc_neighbor_energy(neighbor)
        may_swap_energy = self.calc_neighbor_energy(may_swap_neighbor)
        is_swap = self.MCMC_step(original_energy, may_swap_energy, temp_scale, rng)
        if is_swap:
            return may_swap_neighbor
        else:
            return neighbor

