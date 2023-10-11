nohup python -u run_dt.py --devices 0  > output_dt.log &
nohup python -u run_noise.py --devices 1  > output_noise.log &
nohup python -u run_stateDim.py --devices 2  > output_stateDim.log &