# python main.py --config-name=emb_512_n_layers_5_reg_0.001_gentle_0.1 > /dev/null 2>&1 &
# python main.py --config-name=emb_512_n_layers_5_reg_0.001_gentle_0.05 > /dev/null 2>&1 &

python main.py --config-name=emb_512_n_layers_7_reg_0.001_gentle_0.05 > /dev/null 2>&1 &

# python main.py --config-name=emb_512_n_layers_8  > /dev/null 2>&1 &
python main.py --config-name=emb_512_n_layers_8_reg_0.001_gentle_0.1 > /dev/null 2>&1 &
python main.py --config-name=emb_512_n_layers_8_reg_0.001_gentle_0.05 > /dev/null 2>&1 &
python main.py --config-name=emb_512_n_layers_8_reg_0.001 > /dev/null 2>&1 &

wait
echo "Finished training all the models"