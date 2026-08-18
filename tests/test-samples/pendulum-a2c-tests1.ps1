$model_file = 'models/pendulum-a2c'
$history_file = 'graphics/pendulum-a2c'
$animation_file = 'graphics/pendulum-a2c'
$paramstr = "gaussian"

$device = 'cpu'
$env:RINDOW_NEURALNETWORKS_BACKEND = ""

$env:RL_SEED = "42"
echo "Running Pendulum A2C with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $paramstr + '-seed' + $env:RL_SEED + '-' + $device + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $paramstr + '-' + 'history' + '-seed' + $env:RL_SEED + '-' + $device + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $paramstr + '-' + 'animation' + '-seed' + $env:RL_SEED + '-' + $device + '.gif'
$log_file = "logs/pendulum-a2c" + "-" + $paramstr + "-seed" + $env:RL_SEED + "-" + $device + ".log"
php samples/pendulum-a2c.php > $log_file

$env:RL_SEED = "123"
echo "Running Pendulum A2C with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $paramstr + '-seed' + $env:RL_SEED + '-' + $device + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $paramstr + '-' + 'history' + '-seed' + $env:RL_SEED + '-' + $device + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $paramstr + '-' + 'animation' + '-seed' + $env:RL_SEED + '-' + $device + '.gif'
$log_file = "logs/pendulum-a2c" + "-" + $paramstr + "-seed" + $env:RL_SEED + "-" + $device + ".log"
php samples/pendulum-a2c.php > $log_file

$env:RL_SEED = "1234"
echo "Running Pendulum A2C with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $paramstr + '-seed' + $env:RL_SEED + '-' + $device + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $paramstr + '-' + 'history' + '-seed' + $env:RL_SEED + '-' + $device + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $paramstr + '-' + 'animation' + '-seed' + $env:RL_SEED + '-' + $device + '.gif'
$log_file = "logs/pendulum-a2c" + "-" + $paramstr + "-seed" + $env:RL_SEED + "-" + $device + ".log"
php samples/pendulum-a2c.php > $log_file

$device = 'gpu'
$env:RINDOW_NEURALNETWORKS_BACKEND = "rindowclblast::GPU"

$env:RL_SEED = "42"
echo "Running Pendulum A2C with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $paramstr + '-seed' + $env:RL_SEED + '-' + $device + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $paramstr + '-' + 'history' + '-seed' + $env:RL_SEED + '-' + $device + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $paramstr + '-' + 'animation' + '-seed' + $env:RL_SEED + '-' + $device + '.gif'
$log_file = "logs/pendulum-a2c" + "-" + $paramstr + "-seed" + $env:RL_SEED + "-" + $device + ".log"
php samples/pendulum-a2c.php > $log_file

$env:RL_SEED = "123"
echo "Running Pendulum A2C with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $paramstr + '-seed' + $env:RL_SEED + '-' + $device + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $paramstr + '-' + 'history' + '-seed' + $env:RL_SEED + '-' + $device + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $paramstr + '-' + 'animation' + '-seed' + $env:RL_SEED + '-' + $device + '.gif'
$log_file = "logs/pendulum-a2c" + "-" + $paramstr + "-seed" + $env:RL_SEED + "-" + $device + ".log"
php samples/pendulum-a2c.php > $log_file

$env:RL_SEED = "1234"
echo "Running Pendulum A2C with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $paramstr + '-seed' + $env:RL_SEED + '-' + $device + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $paramstr + '-' + 'history' + '-seed' + $env:RL_SEED + '-' + $device + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $paramstr + '-' + 'animation' + '-seed' + $env:RL_SEED + '-' + $device + '.gif'
$log_file = "logs/pendulum-a2c" + "-" + $paramstr + "-seed" + $env:RL_SEED + "-" + $device + ".log"
php samples/pendulum-a2c.php > $log_file
