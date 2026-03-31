import matplotlib.pyplot as plt
import numpy as np

# N_grid: total_memory (GB)
mem_usage_grid_8 = {
    "1048576": 10.37,
    "131072": 4.86,
    "16384": 1.43,
    "2048": 0.84,
    "256": 0.81,

    }

mem_usage_grid_4 = {
    "1048576": 8.74,
    "131072": 1.81,
    "16384": 0.90,
    #"2048": 0.44,
    #"256": 0.41,
}

mem_usage_grid_16 = {
    "8338608": 59.52,
    "33554432": 180.84,
}

dict_lis = [mem_usage_grid_8, mem_usage_grid_4, mem_usage_grid_16]

fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.set_xscale('log')
ax.set_yscale('log')
#ax[1].set_xscale('log')
#ax[1].set_yscale('log')

labels = ["n=8", "n=4", "n=16"]

def get_vals(dict_: dict):
    mem_ls = []
    grid_ls = []
    for key in dict_.keys():
        mem = dict_[key]
        grid = int(key)
        mem_ls.append(mem)
        grid_ls.append(grid)
    
    return np.array(mem_ls), np.array(grid_ls)

mems_16, grids_16 = get_vals(mem_usage_grid_16)
mems_8, grids_8 = get_vals(mem_usage_grid_8)
mems_4, grids_4 = get_vals(mem_usage_grid_4)

#mem_ratio = mems_4/mems_8

ax.scatter(grids_8, mems_8, marker='x', label=labels[0])
ax.scatter(grids_4, mems_4, marker='x', label=labels[1])
ax.scatter(grids_16, mems_16, marker='x', label = labels[2])

ax.legend()

ax.set_xlabel("Number of gridpoints")
ax.set_ylabel("Memory usage (GB)")
ax.set_title("Memory used for different grid sizes and n MPI ranks")

#ax[1].scatter(grids_8, mem_ratio, marker='x', color='green')
#ax[1].set_xlabel("Number of gridpoints")
#ax[1].set_ylabel("Memory usage ratio")
#ax[1].set_title("Memory n=4/Memory n=8")
plt.savefig("outputs/Mem_grid_extended.png")
