@echo off
REM Quest test — starts server for headset connection
echo.
echo ============================================================
echo   SocialSenseAR Server — Quest Mode
echo ============================================================
echo.
echo Your IP addresses:
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do echo   %%a
echo.
echo In Unity SocialSenseClient, set Server URL to:
echo   ws://YOUR_IP:8765
echo.
echo ============================================================
echo.
python -m server.main --device cuda
