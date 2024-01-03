#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH -o test_2.out
#SBATCH -p short
#SBATCH -p batch

#module load anaconda

# Lets create a new folder for our output files
filename="HKO_partial.csv"
out_dir="HKO_file_partial"
python3 ./GP_Inference.py $filename $out_dir

