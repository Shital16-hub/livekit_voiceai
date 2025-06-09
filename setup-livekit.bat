@echo off
echo LiveKit SIP Trunk Setup
echo =======================

REM Set environment variables
set LIVEKIT_URL=wss://voice-ai-0gc026ty.livekit.cloud
set LIVEKIT_API_KEY=APIBkSkKApLrnrS
set LIVEKIT_API_SECRET=PRN97cK0sU5A1Ou22A5dieMPvka7odMgWkZyBFSUUaV

echo Creating LiveKit SIP trunks for +12563968259...

echo.
echo 📞 Creating inbound trunk...
lk.exe sip inbound create configs\inbound-trunk.json

echo.
echo 📞 Creating outbound trunk...
lk.exe sip outbound create configs\outbound-trunk.json

echo.
echo 🎯 Creating dispatch rule...
lk.exe sip dispatch create configs\dispatch-rule.json

echo.
echo 📋 Checking created trunks...
echo Inbound trunks:
lk.exe sip inbound list

echo.
echo Outbound trunks:
lk.exe sip outbound list

echo.
echo Dispatch rules:
lk.exe sip dispatch list

echo.
echo ✅ LiveKit setup complete!
echo Your Voice AI agent is ready for +12563968259
pause