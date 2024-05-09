#!/bin/bash
#SBATCH --job-name=AE
#SBATCH --mem-per-cpu=2G
#SBATCH --partition long
#SBATCH --nodes=1
#SBATCH --time 2-00:00:00
#SBATCH -c 9
#SBATCH --gres=gpu:1
#SBATCH --account research
# SBATCH -w gnode077

mv slurm-* ./outputs
CURRENT_DATE=`date +"%A, %b %d %Y, %I:%M:%S %p"`

echo "<===========> Details <===========>"
echo "Gnode: $(hostname)"
echo "User: ${USER}"
echo "Date: ${CURRENT_DATE}"
echo "<==================================>"

source activate cong
echo "env activated"

gpustat 

echo "<======> Scratch Directory <=======>"
cd /scratch
if [ ! -d "fk" ]; then
    echo "fk not present, creating it ..."
    mkdir fk
    echo "fk created"
else
    echo "fk present"
fi

cd fk
echo "<======> Dataset <=======>"
if [ ! -d "R1" ]; then
    echo "dataset not present, copying it ..."
    rcp md.faizal@ada.iiit.ac.in:/share1/md.faizal/mgm-data.zip ./
    echo "dataset copied"
    unzip mgm-data.zip -q 
    rm mgm-data.zip
    echo "dataset unzipped"
else
    echo "dataset already present"
fi

cd /home2/md.faizal/code/MGM-Spring-2024/Project/

echo "<======> Running code <=======>"
echo ""

python3 main_ae.py

