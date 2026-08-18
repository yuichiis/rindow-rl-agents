$model_file = 'models/cartpole-dqn'
$history_file = 'graphics/cartpole-dqn'
$animation_file = 'graphics/cartpole-dqn'
$device = 'cpu'
$env:RINDOW_NEURALNETWORKS_BACKEND = ""

$paramstr = "ddqn"

$env:RL_SEED = "42"
echo "Running CartPole DQN with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $paramstr + '-seed' + $env:RL_SEED + '-' + $device + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $paramstr + '-' + 'history' + '-seed' + $env:RL_SEED + '-' + $device + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $paramstr + '-' + 'animation' + '-seed' + $env:RL_SEED + '-' + $device + '.gif'
$log_file = "logs/cartpole-dqn" + "-" + $paramstr + "-seed" + $env:RL_SEED + "-" + $device + ".log"
php samples/cartpole-dqn.php > $log_file

$env:RL_SEED = "123"
echo "Running CartPole DQN with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $paramstr + '-seed' + $env:RL_SEED + '-' + $device + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $paramstr + '-' + 'history' + '-seed' + $env:RL_SEED + '-' + $device + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $paramstr + '-' + 'animation' + '-seed' + $env:RL_SEED + '-' + $device + '.gif'
$log_file = "logs/cartpole-dqn" + "-" + $paramstr + "-seed" + $env:RL_SEED + "-" + $device + ".log"
php samples/cartpole-dqn.php > $log_file

$env:RL_SEED = "1234"
echo "Running CartPole DQN with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $paramstr + '-seed' + $env:RL_SEED + '-' + $device + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $paramstr + '-' + 'history' + '-seed' + $env:RL_SEED + '-' + $device + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $paramstr + '-' + 'animation' + '-seed' + $env:RL_SEED + '-' + $device + '.gif'
$log_file = "logs/cartpole-dqn" + "-" + $paramstr + "-seed" + $env:RL_SEED + "-" + $device + ".log"
php samples/cartpole-dqn.php > $log_file

$device = 'gpu'
$env:RINDOW_NEURALNETWORKS_BACKEND = "rindowclblast::GPU"

$paramstr = "ddqn-potential"

$env:RL_SEED = "42"
echo "Running CartPole DQN with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $paramstr + '-seed' + $env:RL_SEED + '-' + $device + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $paramstr + '-' + 'history' + '-seed' + $env:RL_SEED + '-' + $device + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $paramstr + '-' + 'animation' + '-seed' + $env:RL_SEED + '-' + $device + '.gif'
$log_file = "logs/cartpole-dqn" + "-" + $paramstr + "-seed" + $env:RL_SEED + "-" + $device + ".log"
php samples/cartpole-dqn.php > $log_file

$env:RL_SEED = "123"
echo "Running CartPole DQN with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $paramstr + '-seed' + $env:RL_SEED + '-' + $device + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $paramstr + '-' + 'history' + '-seed' + $env:RL_SEED + '-' + $device + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $paramstr + '-' + 'animation' + '-seed' + $env:RL_SEED + '-' + $device + '.gif'
$log_file = "logs/cartpole-dqn" + "-" + $paramstr + "-seed" + $env:RL_SEED + "-" + $device + ".log"
php samples/cartpole-dqn.php > $log_file

$env:RL_SEED = "1234"
echo "Running CartPole DQN with ${paramstr}-seed${env:RL_SEED} on device: ${device}"
$env:RL_MODEL_FILE = $model_file + '-' + $paramstr + '-seed' + $env:RL_SEED + '-' + $device + '.weights'
$env:RL_HISTORY_FILE = $history_file + '-' + $paramstr + '-' + 'history' + '-seed' + $env:RL_SEED + '-' + $device + '.png'
$env:RL_ANIMATION_FILE = $animation_file + '-' + $paramstr + '-' + 'animation' + '-seed' + $env:RL_SEED + '-' + $device + '.gif'
$log_file = "logs/cartpole-dqn" + "-" + $paramstr + "-seed" + $env:RL_SEED + "-" + $device + ".log"
php samples/cartpole-dqn.php > $log_file
