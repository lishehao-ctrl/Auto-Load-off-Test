# Auto-Load-off-Test
# Instrument Control System

A Python-based GUI system for controlling and measuring instruments (AWG + Oscilloscope).
Developed with `tkinter` and `pyvisa`.

## Features
- Auto frequency sweep and amplitude control
- Live data plotting and calibration
- Config file management
- Real-time AWG/OSC linkage via VISA/LAN

## Folder Structure
- `ui.py` – main GUI
- `deviceMng.py` – device and channel manager
- `equips.py` – instrument drivers
- `test.py` – test control logic
- `configMgr.py` – configuration I/O

## Run
```bash
python main.py
