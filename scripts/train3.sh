echo "##############"
echo "Abnormal"
echo "##############"
python /repos/mrnet/scripts/train3.py $1 -d abnormal --gpu --epochs 40

echo "################"
echo "ACL"
echo "################"
python /repos/mrnet/scripts/train3.py $1 -d acl --gpu --epochs 40

echo "#################"
echo "Meniscus"
echo "#################"
python /repos/mrnet/scripts/train3.py $1 -d meniscus --gpu --epochs 40
