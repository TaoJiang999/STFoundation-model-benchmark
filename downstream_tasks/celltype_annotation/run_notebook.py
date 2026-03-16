import os
import subprocess
import glob
from tqdm import tqdm
import argparse
print(os.getcwd())
# 设置环境
kernel_path = "/home/cavin/anaconda3/envs/tao/bin/python"

parser = argparse.ArgumentParser(description="运行notebook 脚本")
parser.add_argument("--np", type=str, required=True)
notebook_files = []
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