"""
Environment verification script for the ECE 4300 Object Detection project.

Checks that Python and PyTorch are installed correctly and that both the
CPU and Apple Metal Performance Shaders (MPS) backends can run tensor math.

Run from the project root:
    python tests/check_environment.py
"""

import platform
import subprocess
import sys

import torch


def get_processor_name():
    """Return the Mac processor name (e.g. 'Apple M2')."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return platform.processor() or "Unknown"


def check_cpu():
    """Run a small matrix multiplication on the CPU."""
    try:
        a = torch.rand(256, 256, device="cpu")
        b = torch.rand(256, 256, device="cpu")
        c = a @ b
        return True, f"256x256 matmul OK, result shape {tuple(c.shape)}"
    except Exception as e:
        return False, str(e)


def check_mps():
    """Run a small matrix multiplication on the Apple GPU (MPS)."""
    if not torch.backends.mps.is_available():
        return False, "MPS backend not available"
    try:
        a = torch.rand(256, 256, device="mps")
        b = torch.rand(256, 256, device="mps")
        c = a @ b
        torch.mps.synchronize()  # wait for the GPU to finish

        # Compare against the CPU result to confirm correctness
        expected = a.cpu() @ b.cpu()
        if torch.allclose(c.cpu(), expected, atol=1e-3):
            return True, f"256x256 matmul OK on {c.device}, matches CPU result"
        return False, "MPS result does not match CPU result"
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 55)
    print(" ECE 4300 - Environment Check")
    print("=" * 55)

    print(f"Python version      : {platform.python_version()}")
    print(f"Python executable   : {sys.executable}")
    print(f"PyTorch version     : {torch.__version__}")
    print(f"Processor           : {get_processor_name()}")
    print(f"Architecture        : {platform.machine()}")
    print(f"macOS version       : {platform.mac_ver()[0]}")
    print("-" * 55)

    cpu_ok, cpu_msg = check_cpu()
    print(f"CPU execution       : {'PASS' if cpu_ok else 'FAIL'} - {cpu_msg}")

    print(f"MPS built           : {torch.backends.mps.is_built()}")
    print(f"MPS available       : {torch.backends.mps.is_available()}")

    mps_ok, mps_msg = check_mps()
    print(f"MPS tensor test     : {'PASS' if mps_ok else 'FAIL'} - {mps_msg}")
    print("=" * 55)

    # Non-zero exit code if anything failed
    sys.exit(0 if (cpu_ok and mps_ok) else 1)


if __name__ == "__main__":
    main()
