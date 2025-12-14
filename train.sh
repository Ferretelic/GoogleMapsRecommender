python main.py --config-name=emb_512 > /dev/null 2>&1 &
python main.py --config-name=emb_1024 > /dev/null 2>&1 &

python main.py --config-name=emb_256_n_layers_4 > /dev/null 2>&1 &
python main.py --config-name=emb_512_n_layers_4 > /dev/null 2>&1 &
python main.py --config-name=emb_1024_n_layers_4 > /dev/null 2>&1 &

python main.py --config-name=emb_256_reg_0.001 > /dev/null 2>&1 &
python main.py --config-name=emb_512_reg_0.001 > /dev/null 2>&1 &
python main.py --config-name=emb_1024_reg_0.001 > /dev/null 2>&1 &

python main.py --config-name=reg_0.01 > /dev/null 2>&1 &
python main.py --config-name=reg_0.001 > /dev/null 2>&1 &


wait
echo "Finished training all the models"