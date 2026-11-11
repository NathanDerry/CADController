
import network
import socket
import time
import gc
import struct
 
# ==================================
# CONFIGURATION
# ==================================
 
SSID = "Verizon_4J9GBK"
PASSWORD = "dab7-pond-basic"
 
PC_IP = "192.168.1.154"
PORT = 5000
 
WIDTH = 160
HEIGHT = 120
FRAME_SIZE = WIDTH * HEIGHT * 2      # 38,400
HEADER_SIZE = 4
PACKET_SIZE = HEADER_SIZE + FRAME_SIZE
 
# Camera clock. Lower = less background DMA fighting Wi-Fi.
# Start at 10 MHz; raise toward 20 MHz while send time stays low.
XCLK_FREQ = 13_000_000
 
STATS_INTERVAL_MS = 2000
 
# ==================================
# WIFI CONNECTION
# ==================================
 
wifi = network.WLAN(network.STA_IF)
 
wifi.active(False)
time.sleep(1)
wifi.active(True)
time.sleep(1)
 
# On ESP32, PM_NONE == 0 (0xa11140 is a Pico W value)
PM_NONE = getattr(wifi, "PM_NONE", 0)
 
 
def disable_power_save():
    try:
        wifi.config(pm=PM_NONE)
    except Exception as e:
        print("Could not set pm:", e)
 
 
disable_power_save()
 
print("Connecting to Wi-Fi...")
wifi.connect(SSID, PASSWORD)
 
for i in range(40):
    if wifi.isconnected():
        break
    time.sleep_ms(500)
 
if not wifi.isconnected():
    raise RuntimeError("Wi-Fi connection failed")
 
disable_power_save()
 
try:
    pm_now = wifi.config("pm")
except Exception:
    pm_now = "?"
 
print("Connected:", wifi.ifconfig()[0], "| pm:", pm_now)
 
 
def get_rssi():
    try:
        return wifi.status("rssi")
    except Exception:
        return None
 
 
# ==================================
# CAMERA INITIALIZATION
# ==================================
# Low-load settings:
#   - XCLK reduced (less PSRAM DMA traffic)
#   - 1 framebuffer, grab only when asked (no continuous
#     background capture competing with Wi-Fi)
 
print("Initializing camera...")
 
cam = None
 
try:
    from camera import Camera, GrabMode, PixelFormat, FrameSize
 
    cam = Camera(
        pixel_format=PixelFormat.RGB565,
        frame_size=FrameSize.QQVGA,
        xclk_freq=XCLK_FREQ,
        fb_count=1,
        grab_mode=GrabMode.WHEN_EMPTY,
    )
    print("Camera created with low-load settings")
 
except Exception as e:
    print("Low-load camera config failed:", e)
    print("Falling back to default Camera()")
    from camera import Camera
    cam = Camera()
 
# Some firmware auto-inits in the constructor; init() is then harmless
try:
    cam.init()
except Exception as e:
    print("cam.init() skipped:", e)
 
cam.set_vflip(True)
cam.set_hmirror(True)
 
print("Sensor:", cam.get_sensor_name())
print("Pixel format:", cam.get_pixel_format())
 
# Throw away a few frames so exposure settles
for _ in range(3):
    try:
        cam.capture()
    except Exception:
        pass
 
# ==================================
# PREALLOCATED SEND BUFFER
# ==================================
# Header + frame in ONE buffer -> one sendall() per frame.
 
packet = bytearray(PACKET_SIZE)
struct.pack_into("!I", packet, 0, FRAME_SIZE)
packet_mv = memoryview(packet)
frame_slot = packet_mv[HEADER_SIZE:]
 
IPPROTO_TCP = getattr(socket, "IPPROTO_TCP", 6)
TCP_NODELAY = getattr(socket, "TCP_NODELAY", 1)
 
# ==================================
# STREAMING
# ==================================
 
while True:
 
    sock = None
 
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
 
        try:
            sock.setsockopt(IPPROTO_TCP, TCP_NODELAY, 1)
            print("TCP_NODELAY enabled")
        except Exception as e:
            print("TCP_NODELAY unavailable:", e)
 
        print("Connecting to PC...")
        sock.connect((PC_IP, PORT))
        print("TCP connected!")
 
        total_frames = 0
        stat_frames = 0
        stat_capture = 0
        stat_copy = 0
        stat_send = 0
        stat_start = time.ticks_ms()
 
        gc.collect()
 
        while True:
 
            # ---------- CAPTURE ----------
            t0 = time.ticks_ms()
            frame = cam.capture()
            t1 = time.ticks_ms()
 
            if frame is None:
                print("Capture failed")
                continue
 
            if len(frame) != FRAME_SIZE:
                print("Unexpected frame size:", len(frame))
                frame = None
                continue
 
            # ---------- COPY ----------
            frame_slot[:] = frame
            frame = None
            t2 = time.ticks_ms()
 
            # ---------- SEND ----------
            sock.sendall(packet)
            t3 = time.ticks_ms()
 
            # ---------- STATS ----------
            total_frames += 1
            stat_frames += 1
            stat_capture += time.ticks_diff(t1, t0)
            stat_copy += time.ticks_diff(t2, t1)
            stat_send += time.ticks_diff(t3, t2)
 
            elapsed = time.ticks_diff(t3, stat_start)
 
            if elapsed >= STATS_INTERVAL_MS:
                fps = stat_frames * 1000 / elapsed
                mbps = stat_frames * PACKET_SIZE * 8 / elapsed / 1000
 
                print("\n-----------------------")
                print("Frames:", total_frames)
                print("FPS:", round(fps, 2))
                print("Throughput:", round(mbps, 2), "Mbps")
                print("Capture avg:", round(stat_capture / stat_frames, 1), "ms")
                print("Copy avg:", round(stat_copy / stat_frames, 1), "ms")
                print("Send avg:", round(stat_send / stat_frames, 1), "ms")
                print("RSSI:", get_rssi(), "dBm")
                print("Free RAM:", gc.mem_free())
                print("-----------------------")
 
                stat_frames = 0
                stat_capture = 0
                stat_copy = 0
                stat_send = 0
                stat_start = time.ticks_ms()
 
    except OSError as e:
        print("TCP error:", e)
        time.sleep(2)
 
    except Exception as e:
        print("Streaming error:", e)
        time.sleep(2)
 
    finally:
        if sock is not None:
            sock.close()
        gc.collect()
        print("Reconnecting...")