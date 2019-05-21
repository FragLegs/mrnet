echo "##############"
echo "Meniscus Axial"
echo "##############"
python /repos/mrnet/scripts/train.py $1 -d meniscus -s axial --gpu --epochs 40

echo "################"
echo "Meniscus Coronal"
echo "################"
python /repos/mrnet/scripts/train.py $1 -d meniscus -s coronal --gpu --epochs 40

echo "#################"
echo "Meniscus Sagittal"
echo "#################"
python /repos/mrnet/scripts/train.py $1 -d meniscus -s sagittal --gpu --epochs 40
