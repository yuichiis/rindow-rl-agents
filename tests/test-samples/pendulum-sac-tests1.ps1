$model_file = 'models/pendulum-sac'
$history_file = 'graphics/pendulum-sac'
$animation_file = 'graphics/pendulum-sac'
$log_file = 'logs/pendulum-sac'
$php_file = 'samples/pendulum-sac-gsde.php'
$paramstr = "gsde"

$device = 'cpu'
$env:RINDOW_NEURALNETWORKS_BACKEND = ""

$env:RL_SEED = "42"
echo "Running Pendulum SAC with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $paramstr + '-seed' + $env:RL_SEED + '-' + $device + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $paramstr + '-' + 'history' + '-seed' + $env:RL_SEED + '-' + $device + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $paramstr + '-' + 'animation' + '-seed' + $env:RL_SEED + '-' + $device + '.gif'
$log_file_full = $log_file + "-" + $paramstr + "-seed" + $env:RL_SEED + "-" + $device + ".log"
php $php_file > $log_file_full

$env:RL_SEED = "123"
echo "Running Pendulum SAC with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $paramstr + '-seed' + $env:RL_SEED + '-' + $device + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $paramstr + '-' + 'history' + '-seed' + $env:RL_SEED + '-' + $device + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $paramstr + '-' + 'animation' + '-seed' + $env:RL_SEED + '-' + $device + '.gif'
$log_file_full = $log_file + "-" + $paramstr + "-seed" + $env:RL_SEED + "-" + $device + ".log"
php $php_file > $log_file_full

$env:RL_SEED = "1234"
echo "Running Pendulum SAC with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $paramstr + '-seed' + $env:RL_SEED + '-' + $device + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $paramstr + '-' + 'history' + '-seed' + $env:RL_SEED + '-' + $device + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $paramstr + '-' + 'animation' + '-seed' + $env:RL_SEED + '-' + $device + '.gif'
$log_file_full = $log_file + "-" + $paramstr + "-seed" + $env:RL_SEED + "-" + $device + ".log"
php $php_file > $log_file_full

$device = 'gpu'
$env:RINDOW_NEURALNETWORKS_BACKEND = "rindowclblast::GPU"

$env:RL_SEED = "42"
echo "Running Pendulum SAC with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $paramstr + '-seed' + $env:RL_SEED + '-' + $device + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $paramstr + '-' + 'history' + '-seed' + $env:RL_SEED + '-' + $device + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $paramstr + '-' + 'animation' + '-seed' + $env:RL_SEED + '-' + $device + '.gif'
$log_file_full = $log_file + "-" + $paramstr + "-seed" + $env:RL_SEED + "-" + $device + ".log"
php $php_file > $log_file_full

$env:RL_SEED = "123"
echo "Running Pendulum SAC with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $paramstr + '-seed' + $env:RL_SEED + '-' + $device + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $paramstr + '-' + 'history' + '-seed' + $env:RL_SEED + '-' + $device + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $paramstr + '-' + 'animation' + '-seed' + $env:RL_SEED + '-' + $device + '.gif'
$log_file_full = $log_file + "-" + $paramstr + "-seed" + $env:RL_SEED + "-" + $device + ".log"
php $php_file > $log_file_full

$env:RL_SEED = "1234"
echo "Running Pendulum SAC with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $paramstr + '-seed' + $env:RL_SEED + '-' + $device + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $paramstr + '-' + 'history' + '-seed' + $env:RL_SEED + '-' + $device + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $paramstr + '-' + 'animation' + '-seed' + $env:RL_SEED + '-' + $device + '.gif'
$log_file_full = $log_file + "-" + $paramstr + "-seed" + $env:RL_SEED + "-" + $device + ".log"
php $php_file > $log_file_full
