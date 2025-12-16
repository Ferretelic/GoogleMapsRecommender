python main.py --config-name=n_layers_8  > /dev/null 2>&1 &
wait

python main.py --config-name=emb_512_n_layers_7_reg_0.01  > /dev/null 2>&1 &

wait
echo "Finished training all the models"