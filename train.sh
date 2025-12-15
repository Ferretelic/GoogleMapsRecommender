python main.py --config-name=emb_512_n_layers_7_reg_0.001 > /dev/null 2>&1 &
python main.py --config-name=n_layers_7 > /dev/null 2>&1 &

wait
echo "Finished training all the models"