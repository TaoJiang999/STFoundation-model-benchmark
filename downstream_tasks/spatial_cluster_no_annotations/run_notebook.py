import os
import subprocess
import glob
from tqdm import tqdm
print(os.getcwd())
# 设置环境
kernel_path = "/home/cavin/anaconda3/envs/tao/bin/python"
notebook_files = glob.glob("/home/cavin/jt/benchmark/experiments/downstream_tasks/spatial_cluster_no_annotations/*.ipynb")
notebook_files = [notebook_file for notebook_file in notebook_files if not notebook_file.endswith("ominicell.ipynb")]
for notebook in tqdm(notebook_files, desc="Executing notebooks"):
    print(f"Executing {notebook}...")
    
    # 执行 notebook
    execute_command = [
        "jupyter", "nbconvert", "--to", "notebook",
        "--execute", "--inplace", notebook,
        "--ExecutePreprocessor.kernel_name=tao",
    ]
    
    try:
        subprocess.run(execute_command, check=True)
        print(f"Successfully executed {notebook}.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing {notebook}: {e}")
    
    # 清理内存并释放 kernel
    subprocess.run(["jupyter", "notebook", "stop"])

print("All notebooks executed.")