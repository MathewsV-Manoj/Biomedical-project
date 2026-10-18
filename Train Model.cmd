@echo off
cd /d "%~dp0"
matlab -batch "addpath('matlab'); chestinsight_train('MaxEpochs',10);"
pause
