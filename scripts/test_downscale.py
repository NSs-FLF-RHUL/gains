from gains.utils.misc import downsample_h5_file
from pathlib import Path

dir_trial = Path("outputs/larger_r/su_equator")
list_paths = []
for p in dir_trial.rglob("*p*"):
    list_paths.append(p)

for path in list_paths:
    save_dir = Path("outputs/larger_r/downsampled") / path.parent.name 
    Path.mkdir(save_dir, parents=True, exist_ok=True)
    downsample_h5_file(path, save_dir/ path.name)
    print(f"file {path.name} downsampled")

print("=====downsampling done :)========")
