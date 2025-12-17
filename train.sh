python main.py --config-name time_decay_0.001_0.1 > /dev/null 2>&1 &
python main.py --config-name time_decay_0.001_0.3 > /dev/null 2>&1 &
python main.py --config-name time_decay_0.0005_0.1 > /dev/null 2>&1 &
python main.py --config-name time_decay_0.0005_0.3 > /dev/null 2>&1 &
python main.py --config-name time_decay_0.0001_0.1 > /dev/null 2>&1 &
python main.py --config-name time_decay_0.0001_0.3 > /dev/null 2>&1 &

wait
echo "Finished training all the models"