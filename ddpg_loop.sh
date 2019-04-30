
python3 ddpg.py --n-inputs    8 --n-states    32  | tee ddpg_8_32.log
python3 ddpg.py --n-inputs   32 --n-states   256  | tee ddpg_32_256.log
python3 ddpg.py --n-inputs  128 --n-states  1024  | tee ddpg_128_1024.log
