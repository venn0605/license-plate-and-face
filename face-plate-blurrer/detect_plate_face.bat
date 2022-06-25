@echo off

rem yolov5 detect.py
set detect=C:\1_code\face-plate-blurrer\yolov5_blur\detect.py

rem yolov5 weights for detecting plate, name plate.pt
set weights=%cd%\detect-plate-face-weights.pt

rem source dirs to detect
set source=C:\1_code\face-plate-blurrer\Sample_Data

rem folders to export and save pred_labels
set target=C:\1_code\face-plate-blurrer\output

rem script change_labels
set script=%cd%\process2.py

rem if copy and blur
set copy_blur=True

rem env name on your pc
set EnvName=yolov5

call activate %EnvName%

@REM python %detect% --source %source% --weights %weights%  --save-txt --name %target% --nosave --exist-ok

FOR /R %source% %%d in (.) do python %detect% --source %%d --weights %weights% --save-txt --name %target% --nosave --exist-ok

python %script% --source %source% --target %target% --copy %copy_blur% --luv-flag
pause


