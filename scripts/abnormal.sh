echo "##############"
echo "Abnormal Axial"
echo "##############"
python /repos/mrnet/scripts/train.py $1 -d abnormal -s axial --gpu --epochs 40

echo "################"
echo "Abnormal Coronal"
echo "################"
python /repos/mrnet/scripts/train.py $1 -d abnormal -s coronal --gpu --epochs 40

echo "#################"
echo "Abnormal Sagittal"
echo "#################"
python /repos/mrnet/scripts/train.py $1 -d abnormal -s sagittal --gpu --epochs 40
