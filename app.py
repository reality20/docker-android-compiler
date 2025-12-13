from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import sys
import io
import contextlib
import traceback
import multiprocessing
import time
import os
import signal
import asyncio
import subprocess
import sys
import tempfile
import shutil

# Build C Backend before importing quantum_engine
try:
    print("Building C Backend...")
    subprocess.check_call([sys.executable, "build_backend.py", "build_ext", "--inplace"])
    print("C Backend built successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error building C Backend: {e}")
    sys.exit(1)

from quantum_engine import QPU

app = FastAPI()

# Since we flattened the structure, index.html is in root.
# We don't have a static directory anymore for css/js unless we create one.
# But user said "keep all the files in a single folder".
# We can serve index.html directly.

class RunRequest(BaseModel):
    code: str
    num_cores: int
    enable_entanglement: bool = True
    install_libraries: str = ""

class RunResponse(BaseModel):
    output: str
    error: Optional[str] = None
    process_id: Optional[str] = None

# Global store for running processes: process_id -> Process
running_processes = {}

def execute_user_code(code, num_cores, output_queue, enable_entanglement, install_libraries):
    """
    Function to run in a separate process.
    """
    stdout_capture = io.StringIO()
    error_msg = None

    try:
        # Redirect stdout
        with contextlib.redirect_stdout(stdout_capture):
            start_time = time.time()
            from quantum_engine import MPS
            MPS.op_counter = 0

            # Install libraries if requested
            temp_lib_dir = None
            if install_libraries.strip():
                libs = [l.strip() for l in install_libraries.split(',') if l.strip()]
                if libs:
                    print(f"Installing libraries: {', '.join(libs)}...")
                    try:
                        # Create temporary directory for libs
                        temp_lib_dir = tempfile.mkdtemp()
                        sys.path.insert(0, temp_lib_dir)

                        # Install to target dir
                        cmd = [sys.executable, "-m", "pip", "install", "--target", temp_lib_dir] + libs

                        # Run and capture output to stdout (which is redirected to capture)
                        result = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True
                        )
                        print(result.stdout)

                        if result.returncode == 0:
                            print("Libraries installed successfully.")
                        else:
                            print(f"Library installation failed with exit code {result.returncode}")

                    except Exception as e:
                        print(f"Error installing libraries: {e}")

            try:
                # Create QPU
                qpu = QPU(num_cores, enable_entanglement=enable_entanglement)

                # Prepare context
                exec_globals = {
                    "qpu": qpu,
                    "QPU": QPU,
                    "range": range,
                    "len": len,
                    "print": print,
                    "np": sys.modules.get('numpy') # Give access to numpy if needed
                }

                exec(code, exec_globals)

                end_time = time.time()
                duration = end_time - start_time
                ops = MPS.op_counter

                # Calculate rate with SI prefixes
                rate = 0.0
                if duration > 1e-9:
                    rate = ops / duration

                def format_si(n):
                    if n >= 1e9: return f"{n/1e9:.2f}G"
                    if n >= 1e6: return f"{n/1e6:.2f}M"
                    if n >= 1e3: return f"{n/1e3:.2f}k"
                    return f"{n:.2f}"

                print(f"\n--- Execution Stats ---")
                print(f"Duration: {duration:.4f}s")
                print(f"Total Operations: {ops}")
                print(f"Performance: {format_si(rate)} ops/sec")

            finally:
                # Cleanup temp lib dir
                if temp_lib_dir and os.path.exists(temp_lib_dir):
                    shutil.rmtree(temp_lib_dir)

    except Exception:
        traceback.print_exc(file=stdout_capture)
        error_msg = "Execution Error" # Detail is in stdout usually if printed

    # Put result in queue
    output_queue.put((stdout_capture.getvalue(), error_msg))

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/run", response_model=RunResponse)
async def run_code(request: RunRequest):
    # Use multiprocessing
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=execute_user_code,
        args=(request.code, request.num_cores, queue, request.enable_entanglement, request.install_libraries)
    )

    p.start()
    pid = str(p.pid)
    running_processes[pid] = p

    # Wait for completion or until killed?
    # User wants "Kill Process". So we should probably return immediately if it was async,
    # but the current UI expects a response.
    # To support "Kill", we need the UI to poll or we need to wait here but allow interruption.
    # If we wait here, the HTTP request hangs until done.
    # If the user clicks "Kill", they send another request.

    # Let's wait for the process with a timeout or just join.
    # Since we need to return the output, we MUST wait.
    # The "Kill" button would need to send a request to /kill
    # which would terminate this process.
    # But this request handler is blocked on p.join().
    # That's fine, FastAPI is async, but this specific function execution...
    # If we block here, can we handle /kill?
    # Yes, because FastAPI handles requests concurrently (if we use async def, which we do).
    # However, `p.join()` is blocking the event loop if not careful?
    # No, `p.join()` blocks the thread. FastAPI runs async functions in the event loop.
    # If we block the event loop, no other requests.
    # We should run the waiting in a thread pool or use a non-blocking check.

    # Better: Loop with sleep checking is_alive() and allowing context switch.

    try:
        while p.is_alive():
            await asyncio.sleep(0.1)

        # Process finished
        if pid in running_processes:
            del running_processes[pid]

        if not queue.empty():
            output, error = queue.get()
            return RunResponse(output=output, error=error, process_id=pid)
        else:
            return RunResponse(output="Process terminated without output.", error="Terminated", process_id=pid)

    except Exception as e:
        if p.is_alive():
            p.terminate()
        return RunResponse(output="", error=str(e), process_id=pid)

@app.post("/kill")
async def kill_process(process_id: str = None):
    # If process_id is not provided, maybe kill all? Or the last one?
    # For simplicity, we can kill the most recent or accept an ID.
    # But the UI "Kill" button needs to know what to kill.
    # The current request /run blocks until finished, so the UI doesn't have the PID yet!
    # Valid point.

    # Solution: The frontend sends the request. The browser waits.
    # The user clicks "Kill". The browser sends "/kill".
    # But how does "/kill" know which process corresponds to the user's tab?
    # We don't have sessions.
    # Assuming single user or we kill the running process.
    # Since the user says "multiprocessing so it uses all of my resouces",
    # maybe we assume one big job.

    # Let's just kill ALL running processes for now, or the most recent one.

    killed_count = 0
    for pid, p in list(running_processes.items()):
        if p.is_alive():
            p.terminate()
            p.join()
            killed_count += 1
        del running_processes[pid]

    return {"status": "killed", "count": killed_count}

if __name__ == "__main__":
    import uvicorn
    # Use port 7680 as requested
    uvicorn.run(app, host="0.0.0.0", port=7680)
