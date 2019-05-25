echo "##############"
echo "Testing $1"
echo "##############"
python /repos/mrnet/scripts/train3.py $1 -d abnormal --gpu --epochs 1
