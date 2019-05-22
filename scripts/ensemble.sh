
echo "#################"
echo "# Ensembling    #"
echo "#################"
python /repos/mrnet/scripts/ensemble_train.py $1 evals ensembles

echo "#################"
echo "# Ensemble Eval #"
echo "#################"
python /repos/mrnet/scripts/ensemble_eval.py ensembles/$1_ensembles.pkl ensemble_preds evals
