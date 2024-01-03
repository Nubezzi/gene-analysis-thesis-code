#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH -o test_2.out
#SBATCH -p short
#SBATCH -p batch

#module load anaconda

# Lets create a new folder for our outp ut files
out_dir="act_data"
: '
python3 GP_Inference.py "data/data0.csv" $out_dir
python3 GP_Inference.py "data/data1.csv" $out_dir
python3 GP_Inference.py "data/data2.csv" $out_dir
python3 GP_Inference.py "data/data3.csv" $out_dir
python3 GP_Inference.py "data/data4.csv" $out_dir
python3 GP_Inference.py "data/data5.csv" $out_dir
python3 GP_Inference.py "data/data6.csv" $out_dir
python3 GP_Inference.py "data/data7.csv" $out_dir
python3 GP_Inference.py "data/data8.csv" $out_dir
python3 GP_Inference.py "data/data9.csv" $out_dir
python3 GP_Inference.py "data/data10.csv" $out_dir
python3 GP_Inference.py "data/data11.csv" $out_dir 
python3 GP_Inference.py "data/data12.csv" $out_dir 
python3 GP_Inference.py "data/data13.csv" $out_dir 
python3 GP_Inference.py "data/data14.csv" $out_dir 
python3 GP_Inference.py "data/data15.csv" $out_dir 
python3 GP_Inference.py "data/data16.csv" $out_dir 
python3 GP_Inference.py "data/data17.csv" $out_dir 
python3 GP_Inference.py "data/data18.csv" $out_dir 
python3 GP_Inference.py "data/data19.csv" $out_dir 
python3 GP_Inference.py "data/data20.csv" $out_dir 
python3 GP_Inference.py "data/data21.csv" $out_dir 
python3 GP_Inference.py "data/data22.csv" $out_dir 
python3 GP_Inference.py "data/data23.csv" $out_dir 
python3 GP_Inference.py "data/data24.csv" $out_dir 
python3 GP_Inference.py "data/data25.csv" $out_dir 
python3 GP_Inference.py "data/data26.csv" $out_dir 
python3 GP_Inference.py "data/data27.csv" $out_dir 
python3 GP_Inference.py "data/data28.csv" $out_dir 
python3 GP_Inference.py "data/data29.csv" $out_dir 
python3 GP_Inference.py "data/data30.csv" $out_dir 
python3 GP_Inference.py "data/data31.csv" $out_dir 
python3 GP_Inference.py "data/data32.csv" $out_dir 

'
python3 GP_Inference.py "data/data33.csv" $out_dir 
python3 GP_Inference.py "data/data34.csv" $out_dir 
python3 GP_Inference.py "data/data35.csv" $out_dir 
python3 GP_Inference.py "data/data36.csv" $out_dir 
python3 GP_Inference.py "data/data37.csv" $out_dir 
python3 GP_Inference.py "data/data38.csv" $out_dir 
python3 GP_Inference.py "data/data39.csv" $out_dir 
python3 GP_Inference.py "data/data40.csv" $out_dir 


