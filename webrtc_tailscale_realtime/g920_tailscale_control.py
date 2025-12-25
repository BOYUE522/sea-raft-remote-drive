#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
import tkinter as tk
from tkinter import ttk

import roslibpy
from inputs import devices

# ================== Tailscale Config ==================
# Update this to the car's Tailscale IP (run: tailscale ip on car).
CAR_TAILSCALE_IP = "100.65.223.76"
CAR_ROSBRIDGE_PORT = 9090

# ------------------ GEAR CONSTANTS ------------------
GEAR_PARK = 1
GEAR_REVERSE = 2
GEAR_NEUTRAL = 3
GEAR_DRIVE = 4
GEAR_LABELS = {
    GEAR_PARK: "P",
    GEAR_REVERSE: "R",
    GEAR_NEUTRAL: "N",
    GEAR_DRIVE: "D",
}

# ------------------ BUTTON BINDING ------------------
BUTTON_A = "BTN_TRIGGER"   # DRIVE
BUTTON_B = "BTN_THUMB"     # REVERSE
BUTTON_X = "BTN_THUMB2"    # PARK
BUTTON_Y = "BTN_TOP"       # NEUTRAL

# ------------------ NORMALIZATION ------------------
def normalize_throttle(val):  # ABS_Y
    percent = ((255 - val) / 255.0) * 100.0
    amplified = percent * 0.2
    return max(0.0, min(amplified, 100.0))


def normalize_brake(val):  # ABS_Z
    percent = ((255 - val) / 255.0) * 100.0
    amplified = percent * 0.4
    return max(0.0, min(amplified, 100.0))


def normalize_steering(value):  # ABS_X
    return -(value - 32767.5) / 32767.5 * 450.0


# ------------------ GUI ------------------
gui = tk.Tk()
gui.title("G920 Control Status (Tailscale -> Car ROS)")
gui.geometry("400x200")

style = ttk.Style()
style.theme_use("clam")
style.configure("TProgressbar", thickness=20)

gear_var = tk.StringVar(value="P")
throttle_var = tk.DoubleVar(value=0.0)
brake_var = tk.DoubleVar(value=0.0)
steer_var = tk.StringVar(value="0.0 deg")

gear_label = ttk.Label(gui, textvariable=gear_var, font=("Arial", 32))
gear_label.pack(pady=10)

throttle_bar = ttk.Progressbar(gui, variable=throttle_var, maximum=100)
throttle_bar.pack(fill="x", padx=20)

brake_bar = ttk.Progressbar(gui, variable=brake_var, maximum=100)
brake_bar.pack(fill="x", padx=20, pady=(5, 0))

steer_label = ttk.Label(gui, textvariable=steer_var, font=("Arial", 12))
steer_label.pack(pady=5)


# ------------------ ROS (Tailscale) ------------------
def create_ros_connection():
    return roslibpy.Ros(host=CAR_TAILSCALE_IP, port=CAR_ROSBRIDGE_PORT)


ros = create_ros_connection()


def ros_connect_blocking():
    while not ros.is_connected:
        try:
            print(f"Connecting to car ROS: ws://{CAR_TAILSCALE_IP}:{CAR_ROSBRIDGE_PORT}")
            ros.run()
            if ros.is_connected:
                print("Connected to car ROS via Tailscale.")
                break
        except Exception as exc:
            print(f"Connect failed: {exc}")
        print("Retry in 3s...")
        time.sleep(3)


threading.Thread(target=ros_connect_blocking, daemon=True).start()


gear_pub = roslibpy.Topic(ros, "/vehicle/gear/cmd", "ds_dbw_msgs/msg/GearCmd")
throttle_pub = roslibpy.Topic(ros, "/vehicle/throttle/cmd", "ds_dbw_msgs/msg/ThrottleCmd")
brake_pub = roslibpy.Topic(ros, "/vehicle/brake/cmd", "ds_dbw_msgs/msg/BrakeCmd")
steer_pub = roslibpy.Topic(ros, "/vehicle/steering/cmd", "ds_dbw_msgs/msg/SteeringCmd")
enable_pub = roslibpy.Topic(ros, "/vehicle/enable", "std_msgs/msg/Empty")


def enable_dbw_when_ready():
    while not ros.is_connected:
        time.sleep(0.5)
    enable_pub.publish(roslibpy.Message({}))
    print("DBW enable sent.")


threading.Thread(target=enable_dbw_when_ready, daemon=True).start()


# ------------------ Device ------------------
if not devices.gamepads:
    print("G920 not detected.")
    raise SystemExit(1)

gamepad = devices.gamepads[0]

last_gear = GEAR_PARK
current_cmd = {
    "throttle": 0.0,
    "brake": 0.0,
    "steer": 0.0,
}


# ------------------ Command Publisher (100Hz) ------------------
def command_publisher():
    def loop():
        while True:
            if ros.is_connected:
                throttle_msg = {"cmd": current_cmd["throttle"], "cmd_type": 14, "enable": True}
                brake_msg = {"cmd": current_cmd["brake"], "cmd_type": 14, "enable": True}
                steer_msg = {
                    "cmd": current_cmd["steer"],
                    "cmd_type": 2,
                    "enable": True,
                    "clear": False,
                    "ignore": False,
                }
                throttle_pub.publish(roslibpy.Message(throttle_msg))
                brake_pub.publish(roslibpy.Message(brake_msg))
                steer_pub.publish(roslibpy.Message(steer_msg))
            else:
                print("ROS not connected; skipping DBW commands.")
            time.sleep(0.05)

    threading.Thread(target=loop, daemon=True).start()


# ------------------ Brake Zero Check ------------------
def brake_check_loop():
    def loop():
        while True:
            if ros.is_connected and current_cmd["brake"] == 0.0:
                brake_msg = {"cmd": 0.0, "cmd_type": 14, "enable": True}
                for _ in range(10):
                    brake_pub.publish(roslibpy.Message(brake_msg))
                    time.sleep(0.01)
            time.sleep(1.0)

    threading.Thread(target=loop, daemon=True).start()


# ------------------ Gear ------------------
def publish_gear_cmd(gear_value):
    global last_gear
    if gear_value != last_gear and ros.is_connected:
        gear_pub.publish(roslibpy.Message({"cmd": {"value": gear_value}}))
        gear_var.set(GEAR_LABELS[gear_value])
        last_gear = gear_value
    elif not ros.is_connected:
        print("ROS not connected; ignoring gear change.")


print("Listening to G920 input...")


def update_loop():
    global current_cmd
    try:
        events = gamepad._do_iter()
        for event in events:
            if event.ev_type == "Key" and event.state == 1:
                if event.code == BUTTON_A:
                    publish_gear_cmd(GEAR_DRIVE)
                elif event.code == BUTTON_B:
                    publish_gear_cmd(GEAR_REVERSE)
                elif event.code == BUTTON_X:
                    publish_gear_cmd(GEAR_PARK)
                elif event.code == BUTTON_Y:
                    publish_gear_cmd(GEAR_NEUTRAL)
            elif event.ev_type == "Absolute":
                if event.code == "ABS_Y":
                    val = normalize_throttle(event.state)
                    throttle_var.set(val)
                    current_cmd["throttle"] = val
                elif event.code == "ABS_Z":
                    val = normalize_brake(event.state)
                    brake_var.set(val)
                    current_cmd["brake"] = val
                elif event.code == "ABS_X":
                    val = normalize_steering(event.state)
                    steer_var.set(f"{val:.1f} deg")
                    current_cmd["steer"] = val
    except Exception as exc:
        print(f"G920 read error: {exc}")

    gui.after(20, update_loop)


command_publisher()
brake_check_loop()

gui.after(3, update_loop)
try:
    gui.mainloop()
except KeyboardInterrupt:
    print("Exiting...")
finally:
    try:
        steer_pub.unadvertise()
        throttle_pub.unadvertise()
        brake_pub.unadvertise()
        gear_pub.unadvertise()
        enable_pub.unadvertise()
    except Exception as exc:
        print(f"Unadvertise failed: {exc}")
    ros.terminate()
    print("G920 control script stopped.")
