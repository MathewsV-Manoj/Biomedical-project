@echo off
cd /d "%~dp0"
matlab -sd "%~dp0matlab" -r "chestinsight_app"
