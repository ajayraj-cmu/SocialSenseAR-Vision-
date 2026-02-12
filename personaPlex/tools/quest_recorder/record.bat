@echo off
title Quest Screen Recorder
cd /d "%~dp0"

if not exist recordings mkdir recordings

echo ==========================================
echo   Quest Screen Recorder
echo ==========================================
echo.

:: Check adb
adb devices 2>nul | findstr "device" >nul
if errorlevel 1 (
    echo ERROR: No Quest found. Check adb connection.
    pause
    exit /b 1
)

echo Starting recording... Press any key to STOP.
echo.

:: Start screenrecord in background
start /b adb shell screenrecord /sdcard/quest_recording.mp4

:: Wait for keypress
pause >nul

:: Stop recording (kill screenrecord on device)
adb shell pkill -2 screenrecord
timeout /t 2 /nobreak >nul

:: Generate filename with timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set stamp=%datetime:~0,8%_%datetime:~8,6%
set outfile=recordings\quest_%stamp%.mp4

:: Pull from device
echo.
echo Pulling recording...
adb pull /sdcard/quest_recording.mp4 "%outfile%"
adb shell rm /sdcard/quest_recording.mp4

echo.
echo Saved: %outfile%
echo.
pause
