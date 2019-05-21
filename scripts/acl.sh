echo "#########"
echo "ACL Axial"
echo "#########"
python /repos/mrnet/scripts/train.py $1 -d acl -s axial --gpu --epochs 40

echo "###########"
echo "ACL Coronal"
echo "###########"
python /repos/mrnet/scripts/train.py $1 -d acl -s coronal --gpu --epochs 40

echo "############"
echo "ACL Sagittal"
echo "############"
python /repos/mrnet/scripts/train.py $1 -d acl -s sagittal --gpu --epochs 40
