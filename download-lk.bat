@echo off
echo Downloading LiveKit CLI...

REM Create configs directory
mkdir configs 2>nul

REM Download and extract LiveKit CLI
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/livekit/livekit-cli/releases/download/v2.4.10/lk_2.4.10_windows_amd64.zip' -OutFile 'lk.zip'"
powershell -Command "Expand-Archive -Path 'lk.zip' -DestinationPath '.' -Force"
del lk.zip

echo LiveKit CLI downloaded!
echo Testing connection...

REM Set environment variables
set LIVEKIT_URL=wss://voice-ai-0gc026ty.livekit.cloud
set LIVEKIT_API_KEY=APIBkSkKApLrnrS
set LIVEKIT_API_SECRET=PRN97cK0sU5A1Ou22A5dieMPvka7odMgWkZyBFSUUaV

REM Test connection
lk.exe room list

echo.
echo ✅ LiveKit CLI is ready!
echo Now run: setup-livekit.bat
pause