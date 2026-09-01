<?php
declare(strict_types=1);

date_default_timezone_set('UTC');
date_default_timezone_set('UTC');
#ini_set('short_open_tag',true);
if(file_exists(__DIR__.'/../vendor/autoload.php')) {
    $loader = require __DIR__.'/../vendor/autoload.php';
} else {
    $loader = require __DIR__.'/init_autoloader.php';
}
$loader->addPsr4('RindowTest\\RL\\Agents\\',__DIR__);
define('RINDOW_RL_TEST_TEMP_DIR',sys_get_temp_dir().DIRECTORY_SEPARATOR.'rindow-rl-agent-tests');
if (!is_dir(RINDOW_RL_TEST_TEMP_DIR)) {
    mkdir(RINDOW_RL_TEST_TEMP_DIR,0777,true);
}
