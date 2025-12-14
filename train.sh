python main.py --config-name=baseline > /dev/null 2>&1 &

python main.py --config-name=batch_4096 > /dev/null 2>&1 &
python main.py --config-name=batch_2048 > /dev/null 2>&1 &
python main.py --config-name=batch_1024 > /dev/null 2>&1 &

python main.py --config-name=reg_0.001 > /dev/null 2>&1 &
python main.py --config-name=reg_0.00001 > /dev/null 2>&1 &

python main.py --config-name=lr_0.01 > /dev/null 2>&1 &
python main.py --config-name=lr_0.0001 > /dev/null 2>&1 &

python main.py --config-name=n_layers_2 > /dev/null 2>&1 &
python main.py --config-name=n_layers_4 > /dev/null 2>&1 &

python main.py --config-name=emb_32 > /dev/null 2>&1 &
python main.py --config-name=emb_128 > /dev/null 2>&1 &
python main.py --config-name=emb_256 > /dev/null 2>&1 &

wait
echo "Finished training all the models"