import base64
from collections import deque
import io
from pathlib import Path
import time
from typing import Any

import flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import matplotlib.pyplot as plt
import numpy as np

import CuPyNLO

app = flask.Flask(__name__)


class Server:
    def __init__(self):
        CORS(app)
        self._sio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    @app.route("/")
    def index():
        return "Hello, World!"
    
    @app.route("/api/dudley_ssfm", methods=["GET"])
    def dudley_ssfm():
        def generate():
            def to_numpy_host(arr: Any) -> np.ndarray:
                getter = getattr(arr, "get", None)
                host_arr = getter() if callable(getter) else arr
                return np.asarray(host_arr)

            step_info = ["Initialize", "Propagate", "Analyze", "Done"]
            step_idx = 0

            dz = 1e-3
            steps = 100
            range1 = np.arange(steps)
            
            center_wavelength_nm = 835.0
            fiber_length = 0.15

            pump_power_mW = 1.0e4 # Peak power
            pump_pulse_length_ps = 28.4e-3

            n_points = 2**13

            while True:
                if step_idx == 0:
                    pulse1 = CuPyNLO.light.DerivedPulses_v2.SechPulse(
                        power=pump_power_mW,
                        t0_ps=pump_pulse_length_ps,
                        center_wavelength_nm=center_wavelength_nm,
                        time_window_ps=10.0,
                        gdd=0,
                        tod=0.0,
                        n=n_points,
                        frep_MHz=100.0,
                        power_is_avg=False
                    )

                    fiber1 = CuPyNLO.media.fibers.fiber_v2.FiberInstance()
                    fiber1.load_from_db(fiber_length, "dudley")

                    evol = CuPyNLO.interactions.FourWaveMixing.SSFM_v2.SSFM(
                        dz=dz,
                        local_error=0.001,
                        use_simple_raman=True
                    )
                    out = {
                        "step": step_info[step_idx],
                        "message": "Initialization complete"
                    }
                    yield f"data: {flask.json.dumps(out)}\n\n"

                elif step_idx == 1:

                    z_pos = np.linspace(0, fiber1.length, steps + 1)
                    delta_z = float(z_pos[1] - z_pos[0])

                    aw = np.zeros((pulse1.n, steps), dtype=complex)
                    at = np.zeros((pulse1.n, steps), dtype=complex)

                    pulse_out = CuPyNLO.light.PulseBase_v2.Pulse()
                    pulse_out.clone_pulse(pulse1)
                    evol.setup_fftw(pulse1, fiber1, 1.0)
                    evol.load_fiber_parameters(pulse1, fiber1, float(z_pos[0]))

                    deque_time = deque()

                    for i in range(steps):
                        start = time.time()
                        aw_step, at_step, pulse_out = evol.propagate_step(
                            step=i,
                            pulse=pulse_out,
                            fiber=fiber1,
                            dz=delta_z,
                            direction=1
                        )

                        # CuPy arrays must be moved explicitly to host before
                        # storing into NumPy buffers used for SSE payloads.
                        aw[:, i] = to_numpy_host(aw_step)
                        at[:, i] = to_numpy_host(at_step)

                        time_taken = time.time() - start
                        deque_time.append(time_taken)

                        out1 = {
                            "step": step_info[step_idx],
                            "message": f"Propagating... Step {i+1}/{steps}",
                            "time_taken": float(time_taken),
                            "average_time": float(np.mean(deque_time)),
                            "estimated_time_remaining": float(np.mean(deque_time) * (steps - i - 1))
                        }
                        yield f"data: {flask.json.dumps(out1)}\n\n"
                    
                    out = {
                        "step": step_info[step_idx],
                        "message": "Propagation complete"
                    }
                    yield f"data: {flask.json.dumps(out)}\n\n"
                elif step_idx == 2:
                    wl = to_numpy_host(pulse_out.wavelength_nm)
                    t_ps = to_numpy_host(pulse_out.T_ps)

                    loWL = 400
                    hiWL = 1400

                    iis = np.logical_and(wl > loWL, wl < hiWL)
                    iisT = np.logical_and(t_ps > -1, t_ps < 5)

                    xW = wl[iis]
                    xT = t_ps[iisT]
                    zW_in = np.transpose(aw)[:, iis]
                    zT_in = np.transpose(at)[:, iisT]
                    zW = 10 * np.log10(np.abs(zW_in) ** 2)
                    zT = 10 * np.log10(np.abs(zT_in) ** 2)
                    
                    z_pos = z_pos[1:]
                    mlIW = np.max(zW)
                    mlIT = np.max(zT)

                    d = fiber1.beta2_to_d(pulse_out)
                    d = to_numpy_host(d)
                    beta = fiber1.beta2(pulse_out)
                    beta = to_numpy_host(beta)

                    fig, ax = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
                    ax[0, 0].plot(wl, d, 'x')
                    ax[0, 0].set_xlabel("Wavelength [nm]")
                    ax[0, 0].set_ylabel("D [ps/nm/km]")
                    ax[0, 0].set_ylim(-400, None)
                    ax[0, 0].set_xlim(loWL, hiWL)

                    ax[0, 1].plot(wl, beta * 1000, 'x')
                    ax[0, 1].set_xlabel("Wavelength [nm]")
                    ax[0, 1].set_ylabel(r"$\beta_2$ [ps$^2$/km]")
                    ax[0, 1].set_ylim(-350, None)
                    ax[0, 1].set_xlim(loWL, hiWL)
                    
                    ax[1, 0].pcolormesh(xW, z_pos, zW, shading='nearest', vmin=mlIW - 40.0, vmax=mlIW)
                    ax[1, 0].set_xlabel("Wavelength [nm]")
                    ax[1, 0].set_ylabel("Distance [m]")
                    ax[1, 0].autoscale(tight=True)
                    ax[1, 0].set_xlim(loWL, hiWL)

                    ax[1, 1].pcolormesh(xT, z_pos, zT, shading='nearest', vmin=mlIT - 40.0, vmax=mlIT)
                    ax[1, 1].set_xlabel("Delay [ps]")
                    ax[1, 1].set_ylabel("Distance [m]")
                    ax[1, 1].autoscale(tight=True)
                    ax[1, 1].set_xlim(-1, 5)

                    buffer = io.BytesIO()

                    fig.savefig(buffer, dpi=100, format="png")

                    ib64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                    plt.close(fig)

                    data = {"wl": wl.tolist(),
                            "d": d.tolist(),
                            "beta": beta.tolist(),
                            "img1": ib64,
                            "step": step_info[step_idx],
                            "message": "Analysis complete"
                            }
                    yield f"data: {flask.json.dumps(data)}\n\n"
                elif step_idx == 3:
                    out = {
                        "step": step_info[step_idx],
                        "message": "All steps complete"
                    }
                    yield f"data: {flask.json.dumps(out)}\n\n"
                step_idx += 1
                if step_idx >= len(step_info):
                    break
        return flask.Response(generate(), mimetype="text/event-stream")
    
if __name__ == "__main__":
    server = Server()
    try:
        server._sio.run(app, debug=True, host="0.0.0.0", port=2048)
    except KeyboardInterrupt:
        if server is not None:
            server._sio.stop()
