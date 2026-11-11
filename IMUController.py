
import adsk.core
import socket
import threading
import time
import math
import traceback

# ==========================================
# SETTINGS
# ==========================================

EVENT_ID = "com.imucontroller.rotationonly.v1"
UDP_PORT = 5005
UPDATE_INTERVAL = 0.025

SENSITIVITY = 1.0
DEADZONE = 0.5
SMOOTHING = 0.30
MODEL_FOLLOW = True

CALIBRATION_SECONDS = 5

# ==========================================
# GLOBALS
# ==========================================

app = None
ui = None
custom_event = None
handlers = []
worker = None

stop_flag = threading.Event()
ready_flag = threading.Event()

calibration_start = None
neutral_q = None
start_camera = None
filtered_q = None
last_countdown = None

# ==========================================
# QUATERNION FUNCTIONS
# ==========================================

def q_normalize(q):
    n = math.sqrt(sum(x*x for x in q))
    if n < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return [x/n for x in q]


def q_multiply(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b

    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ]


def q_inverse(q):
    return [q[0], -q[1], -q[2], -q[3]]


def q_from_euler(roll, pitch, yaw):
    r = math.radians(roll) / 2
    p = math.radians(pitch) / 2
    y = math.radians(yaw) / 2

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    return q_normalize([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy
    ])


def q_rotate(q, v):
    p = [0.0, v[0], v[1], v[2]]

    result = q_multiply(
        q_multiply(q, p),
        q_inverse(q)
    )

    return result[1:]


def q_axis_angle(q):
    q = q_normalize(q)

    if q[0] < 0:
        q = [-x for x in q]

    angle = 2 * math.acos(
        max(-1.0, min(1.0, q[0]))
    )

    s = math.sqrt(max(0.0, 1-q[0]*q[0]))

    if s < 1e-8:
        return [1.0, 0.0, 0.0], 0.0

    return [q[1]/s, q[2]/s, q[3]/s], angle


def q_from_axis_angle(axis, angle):
    n = math.sqrt(sum(x*x for x in axis))

    if n < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]

    h = angle / 2
    s = math.sin(h) / n

    return q_normalize([
        math.cos(h),
        axis[0]*s,
        axis[1]*s,
        axis[2]*s
    ])


def q_slerp(a, b, t):
    a = q_normalize(a)
    b = q_normalize(b)

    d = sum(a[i]*b[i] for i in range(4))

    if d < 0:
        b = [-x for x in b]
        d = -d

    d = max(-1.0, min(1.0, d))

    if d > 0.9995:
        return q_normalize([
            a[i] + t*(b[i]-a[i])
            for i in range(4)
        ])

    theta = math.acos(d)
    s = math.sin(theta)

    return q_normalize([
        math.sin((1-t)*theta)/s*a[i]
        + math.sin(t*theta)/s*b[i]
        for i in range(4)
    ])


# ==========================================
# VECTOR FUNCTIONS
# ==========================================

def dot(a, b):
    return sum(a[i]*b[i] for i in range(3))


def cross(a, b):
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ]


def normalize(v):
    n = math.sqrt(dot(v, v))

    if n < 1e-12:
        return [0.0, 0.0, 0.0]

    return [x/n for x in v]


def combine_axes(x, y, z, v):
    return [
        x[i]*v[0] + y[i]*v[1] + z[i]*v[2]
        for i in range(3)
    ]


# ==========================================
# CAMERA HANDLER
# ==========================================

class CameraHandler(adsk.core.CustomEventHandler):

    def __init__(self):
        super().__init__()

    def notify(self, args):

        global calibration_start
        global neutral_q
        global start_camera
        global filtered_q
        global last_countdown

        if stop_flag.is_set() or not ready_flag.is_set():
            return

        try:
            values = [
                float(v)
                for v in args.additionalInfo.split(",")
            ]

            # ESP32 sends 4 values.
            # The fourth (zoom) is ignored.
            if len(values) < 3:
                return

            roll, pitch, yaw = values[:3]

            current_q = q_from_euler(
                roll, pitch, yaw
            )

            now = time.monotonic()

            # ==================================
            # CALIBRATION
            # ==================================

            if calibration_start is None:
                calibration_start = now

            elapsed = now - calibration_start

            if elapsed < CALIBRATION_SECONDS:

                remaining = math.ceil(
                    CALIBRATION_SECONDS - elapsed
                )

                if remaining != last_countdown:
                    app.log(
                        "Calibration: {} seconds".format(
                            remaining
                        )
                    )
                    last_countdown = remaining

                return

            if neutral_q is None:

                neutral_q = current_q[:]
                start_camera = app.activeViewport.camera
                filtered_q = [1.0, 0.0, 0.0, 0.0]

                app.log(
                    "Calibration complete - rotation enabled"
                )

                return

            # ==================================
            # ROTATION
            # ==================================

            relative_q = q_multiply(
                q_inverse(neutral_q),
                current_q
            )

            axis, angle = q_axis_angle(relative_q)

            dz = math.radians(DEADZONE)

            if angle <= dz:

                relative_q = [
                    1.0, 0.0, 0.0, 0.0
                ]

            else:

                relative_q = q_from_axis_angle(
                    axis,
                    (angle-dz)*SENSITIVITY
                )

            filtered_q = q_slerp(
                filtered_q,
                relative_q,
                SMOOTHING
            )

            # ==================================
            # CAMERA BASIS
            # ==================================

            eye = start_camera.eye
            target = start_camera.target
            up = start_camera.upVector

            offset = [
                eye.x-target.x,
                eye.y-target.y,
                eye.z-target.z
            ]

            forward = normalize([
                -offset[0],
                -offset[1],
                -offset[2]
            ])

            up_axis = normalize([
                up.x, up.y, up.z
            ])

            right = normalize(
                cross(forward, up_axis)
            )

            up_axis = normalize(
                cross(right, forward)
            )

            backward = [-v for v in forward]

            rotation = filtered_q

            if MODEL_FOLLOW:
                rotation = q_inverse(rotation)

            local_offset = [
                dot(offset, right),
                dot(offset, up_axis),
                dot(offset, backward)
            ]

            local_up = [
                dot(up_axis, right),
                dot(up_axis, up_axis),
                dot(up_axis, backward)
            ]

            rotated_offset = q_rotate(
                rotation,
                local_offset
            )

            rotated_up = q_rotate(
                rotation,
                local_up
            )

            new_offset = combine_axes(
                right,
                up_axis,
                backward,
                rotated_offset
            )

            new_up = normalize(
                combine_axes(
                    right,
                    up_axis,
                    backward,
                    rotated_up
                )
            )

            # ==================================
            # UPDATE CAMERA - ROTATION ONLY
            # ==================================

            viewport = app.activeViewport
            camera = viewport.camera

            camera.isFitView = False
            camera.isSmoothTransition = False

            camera.eye = adsk.core.Point3D.create(
                target.x + new_offset[0],
                target.y + new_offset[1],
                target.z + new_offset[2]
            )

            camera.target = adsk.core.Point3D.create(
                target.x,
                target.y,
                target.z
            )

            camera.upVector = adsk.core.Vector3D.create(
                new_up[0],
                new_up[1],
                new_up[2]
            )

            viewport.camera = camera
            viewport.refresh()

        except:
            app.log(
                "CAMERA ERROR:\n"
                + traceback.format_exc()
            )


# ==========================================
# UDP RECEIVER
# ==========================================

def receive_data():

    sock = None

    try:

        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

        sock.settimeout(0.2)

        sock.bind((
            "127.0.0.1",
            UDP_PORT
        ))

        app.log(
            "UDP listening on port {}".format(
                UDP_PORT
            )
        )

        last_update = 0.0

        while not stop_flag.is_set():

            try:

                data, _ = sock.recvfrom(256)

                if not ready_flag.is_set():
                    continue

                message = data.decode(
                    "utf-8",
                    errors="ignore"
                ).strip()

                values = [
                    float(v)
                    for v in message.split(",")
                ]

                # Accept both 4-value and 5-value
                # packets for compatibility.
                if len(values) not in (4, 5):
                    continue

                now = time.monotonic()

                if now - last_update >= UPDATE_INTERVAL:

                    app.fireCustomEvent(
                        EVENT_ID,
                        message
                    )

                    last_update = now

            except socket.timeout:
                continue

            except ValueError:
                continue

    except:
        if app:
            app.log(
                "UDP ERROR:\n"
                + traceback.format_exc()
            )

    finally:
        if sock:
            sock.close()


# ==========================================
# START
# ==========================================

def run(context):

    global app, ui, custom_event, worker
    global calibration_start, neutral_q
    global start_camera, filtered_q
    global last_countdown

    try:

        app = adsk.core.Application.get()
        ui = app.userInterface

        stop_flag.clear()
        ready_flag.clear()

        calibration_start = None
        neutral_q = None
        start_camera = None
        filtered_q = None
        last_countdown = None

        custom_event = app.registerCustomEvent(
            EVENT_ID
        )

        handler = CameraHandler()
        custom_event.add(handler)
        handlers.append(handler)

        worker = threading.Thread(
            target=receive_data,
            daemon=True
        )

        worker.start()

        ui.messageBox(
            "IMU ROTATION CONTROLLER\n\n"
            "Rotation only is enabled.\n"
            "Zoom data is ignored.\n\n"
            "Click OK and hold still for 5 seconds."
        )

        calibration_start = time.monotonic()
        ready_flag.set()

        app.log(
            "Rotation-only controller started"
        )

    except:
        if ui:
            ui.messageBox(
                traceback.format_exc()
            )


# ==========================================
# STOP
# ==========================================

def stop(context):

    global custom_event, worker

    stop_flag.set()
    ready_flag.clear()

    if worker:
        worker.join(timeout=1)
        worker = None

    if custom_event:

        for handler in handlers:
            try:
                custom_event.remove(handler)
            except:
                pass

        handlers.clear()

        try:
            app.unregisterCustomEvent(EVENT_ID)
        except:
            pass

        custom_event = None

    if app:
        app.log(
            "Rotation-only controller stopped"
        )
