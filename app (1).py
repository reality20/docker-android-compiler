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
# This demo showcases all available QPU operations:
# 1-qubit: h, s, t, x, y, z, rx, ry, rz
# 2-qubit: cx, cz, swap
# Measurement: measure(i)
# Reset: reset(i), reset_all()
import random
import math

core = qpu._core_id
print(f"Core {core}: Starting comprehensive demo.")

# 1. State Preparation & Single Qubit Gates
# Iterate through all single qubit gates
for i in range(qpu.num_qubits):
    qpu.x(i)       # X
    qpu.h(i)       # Hadamard
    qpu.s(i)       # S phase
    qpu.t(i)       # T phase
    qpu.y(i)       # Y
    qpu.z(i)       # Z
    qpu.rx(i, 0.1) # RX
    qpu.ry(i, 0.2) # RY
    qpu.rz(i, 0.3) # RZ

# 2. Entanglement & Two Qubit Gates
# Create a chain of entanglement
for i in range(qpu.num_qubits - 1):
    qpu.cx(i, i+1)   # CNOT
    qpu.cz(i, i+1)   # CZ
    qpu.swap(i, i+1) # SWAP
    qpu.swap(i, i+1) # Swap back to restore order

# 3. Measurement & Feedback
# Measure the first qubit
m = qpu.measure(0)
print(f"Core {core}: Measured qubit 0 = {m}")

# Conditional logic
if m == 1:
    qpu.x(1) # Corrective flip on neighbor

# 4. Reset
qpu.reset(0) # Reset qubit 0 to |0>
m_check = qpu.measure(0)
if m_check != 0:
    print(f"Core {core}: Warning - Reset failed (probabilistic?)")

# 5. Global Reset
qpu.reset_all() # Reset entire register to |0...0>

# 6. Random Circuit Loop
# Run a random circuit using all gates
for step in range(100):
    q = random.randint(0, qpu.num_qubits - 2)

    # Randomly select a gate
    op = random.choice(['h', 's', 't', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'cx', 'cz', 'swap'])

    if op in ['rx', 'ry', 'rz']:
        getattr(qpu, op)(q, random.random())
    elif op in ['cx', 'cz', 'swap']:
        getattr(qpu, op)(q, q+1)
    else:
        getattr(qpu, op)(q)

mem_bytes = qpu.memory_usage()
print(f"Core {core}: Finished. Approx memory: {mem_bytes} bytes")
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
            code_input = gr.Code(label="Simulation Code (Python)", value=default_code, language="python", lines=25)

    with gr.Row():
        status_display = gr.Textbox(label="Status", value="Idle", interactive=False)
        time_display = gr.Textbox(label="Elapsed Time", value="0 s", interactive=False)
        gps_display = gr.Textbox(label="Gates/s (Single Core)", value="0", interactive=False)
        mem_display = gr.Textbox(label="Est. Memory", value="0 MB", interactive=False)
        gates_display = gr.Textbox(label="Total Gates Executed", value="0", interactive=False)

    with gr.Row():
        output_display = gr.Code(label="Output Log", value="", language=None, lines=10, interactive=False)

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
