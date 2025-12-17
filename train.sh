python main.py --config-name=geo_distance_20.0_1.0 > /dev/null 2>&1 &
python main.py --config-name=geo_distance_20.0_1.5 > /dev/null 2>&1 &
python main.py --config-name=geo_distance_20.0_2.0 > /dev/null 2>&1 &
python main.py --config-name=geo_distance_20.0_3.0 > /dev/null 2>&1 &
python main.py --config-name=geo_distance_20.0_4.0 > /dev/null 2>&1 &
python main.py --config-name=geo_distance_20.0_5.0 > /dev/null 2>&1 &

wait
echo "Finished training all the models"