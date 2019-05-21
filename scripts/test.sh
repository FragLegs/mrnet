echo "##############"
echo "Testing $1"
echo "##############"
python /repos/mrnet/scripts/train.py $1 -d abnormal -s axial --gpu --epochs 1
