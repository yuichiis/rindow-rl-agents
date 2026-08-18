$model_file = 'models/mountaincar-ppo-shared'
$history_file = 'graphics/mountaincar-ppo-history'
$animation_file = 'graphics/mountaincar-ppo-trained'
$device = 'cpu'
$env:RINDOW_NEURALNETWORKS_BACKEND = ""

$env:RL_ENTROPY_WEIGHT = "0.01"
$env:RL_EPOCHS = "5"
$paramstr = "entropyw0.01-epochs5"

$env:RL_SEED = "42"
echo "Running MountainCar PPO with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $device + '-' + $paramstr + '-seed' + $env:RL_SEED + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $device + '-' + $paramstr + '-seed' + $env:RL_SEED + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $device + '-' + $paramstr + '-seed' + $env:RL_SEED + '.gif'
$log_file = "logs/mountaincar-ppo-" + $device + "-" + $paramstr + "-seed" + $env:RL_SEED + ".log"
php samples/mountaincar-ppo.php > $log_file

$env:RL_SEED = "1234"
echo "Running MountainCar PPO with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $device + '-' + $paramstr + '-seed' + $env:RL_SEED + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $device + '-' + $paramstr + '-seed' + $env:RL_SEED + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $device + '-' + $paramstr + '-seed' + $env:RL_SEED + '.gif'
$log_file = "logs/mountaincar-ppo-" + $device + "-" + $paramstr + "-seed" + $env:RL_SEED + ".log"
php samples/mountaincar-ppo.php > $log_file

$env:RL_SEED = "123"
echo "Running MountainCar PPO with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $device + '-' + $paramstr + '-seed' + $env:RL_SEED + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $device + '-' + $paramstr + '-seed' + $env:RL_SEED + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $device + '-' + $paramstr + '-seed' + $env:RL_SEED + '.gif'
$log_file = "logs/mountaincar-ppo-" + $device + "-" + $paramstr + "-seed" + $env:RL_SEED + ".log"
php samples/mountaincar-ppo.php > $log_file


$device = 'gpu'
$env:RINDOW_NEURALNETWORKS_BACKEND = "rindowclblast::GPU"
$env:RL_ENTROPY_WEIGHT = "0.01"
$env:RL_EPOCHS = "5"
$paramstr = "entropyw0.01-epochs5"


$env:RL_SEED = "42"
echo "Running MountainCar PPO with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $device + '-' + $paramstr + '-seed' + $env:RL_SEED + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $device + '-' + $paramstr + '-seed' + $env:RL_SEED + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $device + '-' + $paramstr + '-seed' + $env:RL_SEED + '.gif'
$log_file = "logs/mountaincar-ppo-" + $device + "-" + $paramstr + "-seed" + $env:RL_SEED + ".log"
php samples/mountaincar-ppo.php > $log_file

$env:RL_SEED = "1234"
echo "Running MountainCar PPO with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $device + '-' + $paramstr + '-seed' + $env:RL_SEED + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $device + '-' + $paramstr + '-seed' + $env:RL_SEED + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $device + '-' + $paramstr + '-seed' + $env:RL_SEED + '.gif'
$log_file = "logs/mountaincar-ppo-" + $device + "-" + $paramstr + "-seed" + $env:RL_SEED + ".log"
php samples/mountaincar-ppo.php > $log_file

$env:RL_SEED = "123"
echo "Running MountainCar PPO with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $device + '-' + $paramstr + '-seed' + $env:RL_SEED + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $device + '-' + $paramstr + '-seed' + $env:RL_SEED + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $device + '-' + $paramstr + '-seed' + $env:RL_SEED + '.gif'
$log_file = "logs/mountaincar-ppo-" + $device + "-" + $paramstr + "-seed" + $env:RL_SEED + ".log"
php samples/mountaincar-ppo.php > $log_file

