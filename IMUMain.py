
from machine import Pin, I2C
import time
import math
import struct

SDA_PIN = 42
SCL_PIN = 41
IMU_ADDR = 0x6A
SAMPLE_MS = 20

WHO_AM_I = 0x0F
CTRL1_XL = 0x10
CTRL2_G = 0x11
CTRL3_C = 0x12
OUTX_L_G = 0x22
OUTX_L_A = 0x28

i2c = I2C(
    0,
    sda=Pin(SDA_PIN),
    scl=Pin(SCL_PIN),
    freq=400000
)

if IMU_ADDR not in i2c.scan():
    IMU_ADDR = 0x6B

if IMU_ADDR not in i2c.scan():
    raise RuntimeError("LSM6DSO not found")

def write_reg(reg, value):
    i2c.writeto_mem(
        IMU_ADDR, reg, bytes([value])
    )

def read_xyz(reg):
    data = i2c.readfrom_mem(IMU_ADDR, reg, 6)
    return struct.unpack("<hhh", data)

print("WHO_AM_I:",
      hex(i2c.readfrom_mem(IMU_ADDR, WHO_AM_I, 1)[0]))

# Block data update and auto-increment
write_reg(CTRL3_C, 0x44)

# Accelerometer 104 Hz, +/-2 g
write_reg(CTRL1_XL, 0x40)

# Gyroscope 104 Hz, +/-250 dps
write_reg(CTRL2_G, 0x40)

time.sleep_ms(300)

def read_imu():
    gx, gy, gz = read_xyz(OUTX_L_G)
    ax, ay, az = read_xyz(OUTX_L_A)

    gyro = (
        gx * 0.00875,
        gy * 0.00875,
        gz * 0.00875
    )

    accel = (
        ax * 0.000061,
        ay * 0.000061,
        az * 0.000061
    )

    return gyro, accel

# Five-second gyro calibration
print("Keep IMU still for 5 seconds")

sx = sy = sz = 0.0
count = 0
start = time.ticks_ms()

while time.ticks_diff(time.ticks_ms(), start) < 5000:
    gyro, accel = read_imu()
    sx += gyro[0]
    sy += gyro[1]
    sz += gyro[2]
    count += 1
    time.sleep_ms(10)

bias = (
    sx / count,
    sy / count,
    sz / count
)

gyro, accel = read_imu()

ax, ay, az = accel

roll = math.degrees(math.atan2(ay, az))
pitch = math.degrees(
    math.atan2(-ax, math.sqrt(ay*ay + az*az))
)
yaw = 0.0

roll_zero = roll
pitch_zero = pitch

last = time.ticks_ms()

print("Calibration complete")
print("Streaming yaw,pitch,roll")

while True:
    now = time.ticks_ms()
    dt = time.ticks_diff(now, last) / 1000.0
    last = now

    gyro, accel = read_imu()
    gx, gy, gz = gyro
    ax, ay, az = accel

    gx -= bias[0]
    gy -= bias[1]
    gz -= bias[2]

    accel_roll = math.degrees(math.atan2(ay, az))
    accel_pitch = math.degrees(
        math.atan2(-ax, math.sqrt(ay*ay + az*az))
    )

    alpha = 0.98

    roll = alpha * (roll + gx*dt) + \
           (1-alpha) * accel_roll

    pitch = alpha * (pitch + gy*dt) + \
            (1-alpha) * accel_pitch

    yaw += gz * dt

    print("{:.2f},{:.2f},{:.2f}".format(
        yaw,
        pitch - pitch_zero,
        roll - roll_zero
    ))

    time.sleep_ms(SAMPLE_MS)
