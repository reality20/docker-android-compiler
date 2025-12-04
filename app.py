import gradio as gr
import time
import multiprocessing
from backend import SimulationManager

# Global Manager
manager = SimulationManager()

def start_run(bond_dim, num_qubits, num_cores, code):
    if not code.strip():
        return "No code provided", "0", "0", "0", "0", ""

    try:
        manager.start_simulation(code, int(bond_dim), int(num_qubits), int(num_cores))
        return "Simulation Started", "0", "0", "0", "0", ""
    except Exception as e:
        return f"Error: {e}", "0", "0", "0", "0", ""

def stop_run():
    manager.stop_simulation()
    return "Simulation Stopped"

def update_status():
    status = manager.get_status()

    state_str = "Running" if manager.is_running() else "Idle"
    if status['errors']:
        state_str = f"Error: {status['errors'][0]}"

    elapsed = f"{status['elapsed']:.2f} s"
    gps_core = f"{status['gates_per_sec_core']:.2e}"
    mem = f"{status['total_memory'] / 1024**2:.2f} MB"
    total_gates = f"{status['total_gates']}"
    output_log = status['output']

    return state_str, elapsed, gps_core, mem, total_gates, output_log

default_code = """# Example Simulation
# 'qpu' object is available with methods:
# 1-qubit: h, s, t, x, y, z, rx, ry, rz
# 2-qubit: cx, cz, swap
import random

print(f"Starting simulation on {qpu._core_id}...")

# Apply random gates using all available operations
for i in range(1000):
    q = random.randint(0, qpu.num_qubits - 2)

    # Single qubit gates
    qpu.h(q)
    qpu.s(q)
    qpu.t(q)
    qpu.x(q)
    qpu.y(q)
    qpu.z(q)
    qpu.rx(q, 0.1)
    qpu.ry(q, 0.2)
    qpu.rz(q, 0.3)

    # Two qubit gates
    qpu.cx(q, q+1)
    qpu.cz(q, q+1)
    qpu.swap(q, q+1)

print(f"Core {qpu._core_id} finished.")
"""

with gr.Blocks(title="QPU Simulator") as demo:
    gr.Markdown("# QPU Simulator (MPS Backend)")

    with gr.Row():
        with gr.Column(scale=1):
            bond_dim = gr.Number(label="Bond Dimension", value=16, precision=0)
            num_qubits = gr.Number(label="Number of Qubits", value=10, precision=0)
            # Default to all cores
            default_cores = multiprocessing.cpu_count()
            num_cores = gr.Number(label="Number of Cores", value=default_cores, precision=0)

            run_btn = gr.Button("Run Simulation", variant="primary")
            kill_btn = gr.Button("Kill Simulation", variant="stop")

        with gr.Column(scale=2):
            code_input = gr.Code(label="Simulation Code (Python)", value=default_code, language="python", lines=20)

    with gr.Row():
        status_display = gr.Textbox(label="Status", value="Idle", interactive=False)
        time_display = gr.Textbox(label="Elapsed Time", value="0 s", interactive=False)
        gps_display = gr.Textbox(label="Gates/s (Single Core)", value="0", interactive=False)
        mem_display = gr.Textbox(label="Est. Memory", value="0 MB", interactive=False)
        gates_display = gr.Textbox(label="Total Gates Executed", value="0", interactive=False)

    with gr.Row():
        output_display = gr.Code(label="Output Log", value="", language="text", lines=10, interactive=False)

    # Event handlers
    run_btn.click(
        fn=start_run,
        inputs=[bond_dim, num_qubits, num_cores, code_input],
        outputs=[status_display, time_display, gps_display, mem_display, gates_display, output_display]
    )

    kill_btn.click(
        fn=stop_run,
        inputs=[],
        outputs=[status_display]
    )

    # Timer for updates
    timer = gr.Timer(0.5)
    timer.tick(update_status, inputs=[], outputs=[status_display, time_display, gps_display, mem_display, gates_display, output_display])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
