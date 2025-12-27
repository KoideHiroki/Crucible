import numpy as np
from impl.molecule import Soap, Water, Air, MCMCUtl
from impl.molecule import MoleculeKind
import sys

class Tank:
    def __init__(self, soap_ratio, water_ratio, temp_scale, tank_size=100, seed=0, restart=None):
        assert soap_ratio+water_ratio <= 1.0
        self.rng = np.random.default_rng(seed)
        self.temp_scale = temp_scale
        self.tank_size = tank_size
        self.mols = self.init_mols(soap_ratio, water_ratio, restart)

    def init_mols(self, soap_ratio, water_ratio, restart):
        if restart is not None:
            def encode_raw_mols(raw_mol):
                if raw_mol[0] == 1:
                    return np.asarray([MoleculeKind.SoapKind, raw_mol[1]])
                elif raw_mol[0] == 2:
                    return np.asarray([MoleculeKind.WaterKind, raw_mol[1]])
                elif raw_mol[0] == 3:
                    return np.asarray([MoleculeKind.AirKind, raw_mol[1]])
                else:
                    print("invalid restart MolecleKind.")
                    sys.exit()
            mols = np.apply_along_axis(encode_raw_mols, 2, restart)
            W, H, _ = mols.shape
            assert W == self.tank_size and H == self.tank_size
            return mols

        soap_num = int(self.tank_size*self.tank_size*soap_ratio)
        water_num = int(self.tank_size*self.tank_size*water_ratio)
        soaps = [Soap(rng=self.rng) for _ in range(soap_num)]
        waters = [Water() for _ in range(water_num)]
        airs = [Air() for _ in range(self.tank_size*self.tank_size-(soap_num+water_num))]
        mols = soaps + waters + airs
        self.rng.shuffle(mols)
        mols = np.asarray([m.encode() for m in mols])
        mols = mols.reshape(self.tank_size, self.tank_size, 2)
        #print(mols.shape)
        return mols

    def run(self, loop_num, out_prefix, save_step_num):
        for loop_idx in range(loop_num):
            self.step()
            if loop_idx % save_step_num == 0:
                print(loop_idx)
                self.write_log(out_prefix, loop_idx)

    def step(self):
        for row_idx in range(self.tank_size):
            for col_idx in range(self.tank_size):
                if self.rng.random() < 0.1:
                    self.try_swap_7x7(row_idx, col_idx)

    def try_swap(self, row_idx, col_idx):
        neighbor = self.get_neighbor(row_idx, col_idx)
        #print(neighbor.shape)
        mcmc_utl = MCMCUtl()
        new_neighbor = mcmc_utl.try_local_swap(neighbor, self.temp_scale, self.rng)
        self.embed_neighbor(new_neighbor, row_idx, col_idx)

    def try_swap_7x7(self, row_idx, col_idx):

        neighbor = self.get_neighbor_7x7(row_idx, col_idx)
        mcmc_utl = MCMCUtl()

        # まず並進swap
        new_neighbor = mcmc_utl.try_local_swap_7x7(neighbor, self.temp_scale, self.rng)

        # 次に回転（回転は毎回でもいいし、確率で間引いてもOK）
        # 例：毎回回す
        new_neighbor = mcmc_utl.try_local_rotate_7x7(new_neighbor, self.temp_scale, self.rng)

        self.embed_neighbor_7x7(new_neighbor, row_idx, col_idx)

    def get_neighbor_7x7(self, row_idx, col_idx):
        idx = [(row_idx + di) % self.tank_size for di in range(-3, 4)]  # -3..3
        jdx = [(col_idx + dj) % self.tank_size for dj in range(-3, 4)]  # -3..3
        return self.mols[np.ix_(idx, jdx, [0, 1])]

    def get_neighbor(self, row_idx, col_idx):
        idx = [(row_idx + di) % self.tank_size for di in range(-2, 3)]
        jdx = [(col_idx + dj) % self.tank_size for dj in range(-2, 3)]
        return self.mols[np.ix_(idx, jdx, [0, 1])]

    def embed_neighbor_7x7(self, new_neighbor, row_idx, col_idx):
        idx = [(row_idx + di) % self.tank_size for di in range(-3, 4)]
        jdx = [(col_idx + dj) % self.tank_size for dj in range(-3, 4)]
        self.mols[np.ix_(idx, jdx, [0, 1])] = new_neighbor

    def embed_neighbor(self, new_neighbor, row_idx, col_idx):
        idx = [(row_idx + di) % self.tank_size for di in range(-2, 3)]
        jdx = [(col_idx + dj) % self.tank_size for dj in range(-2, 3)]
        self.mols[np.ix_(idx, jdx, [0, 1])] = new_neighbor

    def write_log(self, out_prefix, loop_idx):
        arr = np.apply_along_axis(lambda x: [x[0].value, x[1]], 2, self.mols)
        np.save("./log/"+out_prefix+"_step_{}".format(loop_idx), arr)