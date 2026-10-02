; Demo G-code — 50mm calibration cube trace
; Generated for MuJoCo CoreXY printer simulation demo
; Traces around the 5cm test cube, then prints on its top surface.
; Units: mm
;
; --- Startup ---
M140 S60         ; Set bed temperature 60°C
M104 S215        ; Set hotend temperature 215°C
G28              ; Home all axes
G1 Z5 F600       ; Lift nozzle 5mm
G1 X100 Y100 F6000 ; Move to safe location near cube
G1 Z0.2 F600     ; Lower to first-layer height

; === Skirt Layer (Z = 0.2 mm) ===
; Trace a 54x54mm skirt around the base of the 50mm cube
G1 X101 Y101 F3000     ; Move to front-left corner of skirt
G1 X155 Y101 E1.0 F1500 ; Bottom edge
G1 X155 Y155 E1.0      ; Right edge
G1 X101 Y155 E1.0      ; Top edge
G1 X101 Y101 E1.0      ; Left edge (close)

; === Lift to Top of Cube ===
; Cube is 50mm tall. We lift just above it to print the top layers.
G1 Z50.2 F600          ; Lift nozzle to Z=50.2 (0.2mm layer height on top)

; === Top Layer 1 (Z = 50.2 mm) ===
; Trace exactly on the top edge of the 50x50mm cube
G1 X103 Y103 F3000     ; Move to front-left corner of cube top
G1 X153 Y103 E1.0 F1500 ; Bottom edge
G1 X153 Y153 E1.0      ; Right edge
G1 X103 Y153 E1.0      ; Top edge
G1 X103 Y103 E1.0      ; Left edge (close)

; === Top Layer 2 (Z = 50.4 mm) ===
G1 Z50.4 F600
G1 X103 Y103 F3000
G1 X153 Y103 E1.0 F1500
G1 X153 Y153 E1.0
G1 X103 Y153 E1.0
G1 X103 Y103 E1.0

; === Top Layer 3 (Z = 50.6 mm) ===
G1 Z50.6 F600
G1 X103 Y103 F3000
G1 X153 Y103 E1.0 F1800
G1 X153 Y153 E1.0
G1 X103 Y153 E1.0
G1 X103 Y103 E1.0

; --- Finish ---
G1 Z60 F600      ; Lift nozzle away
G1 X0 Y0 F6000   ; Home XY
M140 S0           ; Bed heater off
M104 S0           ; Hotend heater off
