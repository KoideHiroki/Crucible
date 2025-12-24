import numpy as np
import random
from molecule import Soap, Water, Air, MCMCUtl

class Tank:
    def __init__(self, soap_ratio, water_ratio, temp_scale, tank_size=100, seed=0, ):
        assert soap_ratio+water_ratio <= 1.0
        self.rng = np.random.default_rng(seed)
        self.temp_scale = temp_scale
        self.tank_size = tank_size
        self.mols = self.init_mols(soap_ratio, water_ratio)

    def init_mols(self, soap_ratio, water_ratio):
        soap_num = int(self.tank_size*self.tank_size*soap_ratio)
        water_num = int(self.tank_size*self.tank_size*water_ratio)
        soaps = [Soap(rng=self.rng) for _ in range(soap_num)]
        waters = [Water() for _ in range(water_num)]
        airs = [Air() for _ in range(self.tank_size*self.tank_size-(soap_num+water_num))]
        mols = soaps + waters + airs
        ramdom.shuffle(mols)
        mols = np.asarray([m.encode() for m in mols])
        mols = mols.reshape(self.tank_size, self.tank_size, 2)
        #print(mols.shape)
        return mols

    def run(self, loop_num, out_path, save_step_num):
        for loop_idx in range(loop_num):
            self.step()
            if loop_idx % save_step_num == 0:
                self.write_log(out_path)

    def step(self):
        for row_idx in range(self.tank_size):
            for col_idx in range(self.tank_size):
                self.try_swap(row_idx, col_idx)

    def try_swap(self, row_idx, col_idx):
        neighbor = self.get_neighbor(row_idx, col_idx)
        #print(neighbor.shape)
        new_neighbor = MCMCUtl.try_local_swap(neighbor, self.temp_scale, self.rng)
        self.embed_neighbor(new_neighbor, row_idx, col_idx)

    def get_neighbor(self, row_idx, col_idx):
        return self.mols[row_idx-2:row_idx+2+1, col_idx-2:col_idx+2+1, :]
    
    def embed_neighbor(self, new_neighbor, row_idx, col_idx):
        self.mols[row_idx-2:row_idx+2+1, col_idx-2:col_idx+2+1, :] = new_neighbor