# Tester code to compile, run and compare the output of your CUDA program and sample output
# Usage: python tester.py
#
# The script is set such that the target OS is Linux. If you are using Windows, 
# you will need to change the necessary paths.
#
# The output will show the total number of test cases passed out of 3 sample test cases.

import glob
import os
import subprocess

IN = "/input/test"
OUT = "/output/out"
EXT = ".txt"

CODE_FOLDER = "code/"
TARGET_EXTENSION = ".cu"

def get_code_file():

    script_path = os.path.abspath(__file__)
    absolute_code_folder = os.path.join(os.path.dirname(script_path), CODE_FOLDER)
    search_pattern = os.path.join(absolute_code_folder, f"*{TARGET_EXTENSION}")
    files = glob.glob(search_pattern)

    if len(files) == 0:
        raise FileNotFoundError(f"No .cu file found in {absolute_code_folder}.")
    elif len(files) > 1:
        raise ValueError(f"Found more than one .cu file in {absolute_code_folder}. Expected only one.")

    return os.path.abspath(files[0]), os.path.dirname(script_path)
  
def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = [line.strip() for line in f1.readlines()]
        lines2 = [line.strip() for line in f2.readlines()]

    # if lines1 != lines2:
    #     print(f"Files {file1} and {file2} do not match:")
    #     for i, (line1, line2) in enumerate(zip(lines1, lines2), start=1):
    #         if line1 != line2:
    #             print(f"Line {i}:")
    #             print(f"  {file1}: {line1}")
    #             print(f"  {file2}: {line2}")

    return lines1 == lines2


def compile_and_run_cuda():  
    
    file, folder = get_code_file()
    count = 0
    code_dir = folder + "/" + CODE_FOLDER
    command = f"nvcc {file} -o {code_dir}a.out"
    subprocess.run(command, shell=True)

    for i in range(1, 7):
        print( "Running test case ",i)
        IN1 = folder + IN + str(i) + EXT
        OUT1 = folder + OUT + str(i) + EXT
        run_command = "./a.out < " + IN1
        subprocess.run(run_command, shell=True, cwd=code_dir)

        if(compare_files(OUT1, code_dir + "cuda.out")):
            count = count + 1

    cleanup(code_dir, file)
    print(f"Total number of test cases passed: {count} out of 6")


def cleanup(folder_path, exception_file):

    files = glob.glob(folder_path + "*")
    for file in files:
        if file != exception_file:
            os.remove(file)

def main():
    compile_and_run_cuda()

if __name__ == "__main__":
    main()