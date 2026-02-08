@echo off
REM Local test — starts server + webcam client (no headset needed)
echo Starting server...
start "SocialSenseAR Server" cmd /k python -m server.main --device cuda
timeout /t 5 /nobreak >nul
echo Starting test client...
python -m server.test_client --show
