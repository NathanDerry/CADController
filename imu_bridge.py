
import serial
import socket
import time

SERIAL_PORT = "COM3"
BAUD_RATE = 115200

UDP_HOST = "127.0.0.1"
UDP_PORT = 5005

def main():
    udp = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    while True:
        try:
            print("Connecting to", SERIAL_PORT)

            with serial.Serial(
                SERIAL_PORT,
                BAUD_RATE,
                timeout=1
            ) as ser:

                time.sleep(2)
                ser.reset_input_buffer()

                print("Connected to ESP32")

                while True:
                    line = ser.readline().decode(
                        "utf-8",
                        errors="ignore"
                    ).strip()

                    parts = line.split(",")

                    if len(parts) != 4:
                        continue

                    try:
                        roll, pitch, yaw, zoom = (
                            float(v) for v in parts
                        )
                    except ValueError:
                        continue

                    if zoom not in (-1.0, 0.0, 1.0):
                        continue

                    message = (
                        "{:.3f},{:.3f},{:.3f},{}".format(
                            roll, pitch, yaw, int(zoom)
                        )
                    )

                    udp.sendto(
                        message.encode("utf-8"),
                        (UDP_HOST, UDP_PORT)
                    )

        except serial.SerialException as e:
            print("Serial error:", e)
            print("Retrying in 2 seconds...")
            time.sleep(2)

        except KeyboardInterrupt:
            print("Bridge stopped")
            break

    udp.close()

if __name__ == "__main__":
    main()
