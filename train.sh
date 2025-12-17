python main.py --config-name=geo_distance_5 > /dev/null 2>&1 &
python main.py --config-name=geo_distance_10 > /dev/null 2>&1 &
python main.py --config-name=geo_distance_15 > /dev/null 2>&1 &
python main.py --config-name=geo_distance_20 > /dev/null 2>&1 &

wait

python main.py --config-name=geo_distance_20.0_6.0 > /dev/null 2>&1 &
python main.py --config-name=geo_distance_20.0_7.0 > /dev/null 2>&1 &
python main.py --config-name=geo_distance_20.0_8.0 > /dev/null 2>&1 &

python main.py --config-name=gcl > /dev/null 2>&1 &

wait
echo "Finished training all the models"