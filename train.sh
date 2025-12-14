python main.py --config-name=emb_1024_n_layers_4 > /dev/null 2>&1 &
python main.py --config-name=emb_1024_reg_0.001 > /dev/null 2>&1 &

python main.py --config-name=emb_512_n_layers_4_reg_0.001 > /dev/null 2>&1 &
python main.py --config-name=emb_512_n_layers_5_reg_0.001 > /dev/null 2>&1 &

python main.py --config-name=n_layers_5 > /dev/null 2>&1 &

wait
echo "Finished training all the models"