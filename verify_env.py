"""
AI Outfit Recommendation & Style Matching System
Module 2: Environment Setup & Computer Vision Libraries Verification Script

This script verifies that Python and all required computer vision,
deep learning, and data science libraries are correctly installed and functional.
"""

import sys
import os
import platform

# Ensure UTF-8 output encoding on Windows if supported
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    print(f"\n{CYAN}{'=' * 75}{RESET}")
    print(f"{BOLD}{CYAN}  AI Outfit Recommendation & Style Matching System - Module 2 Setup{RESET}")
    print(f"{CYAN}  Environment & Computer Vision Libraries Diagnostic{RESET}")
    print(f"{CYAN}{'=' * 75}{RESET}\n")


def check_python_environment():
    print(f"{BOLD}[1/4] Checking Python Runtime...{RESET}")
    py_ver = platform.python_version()
    bitness = platform.architecture()[0]
    os_name = platform.system()
    os_release = platform.release()
    print(f"  * Python Version  : {GREEN}{py_ver}{RESET} ({bitness})")
    print(f"  * Executable      : {sys.executable}")
    print(f"  * Operating System: {os_name} {os_release} ({platform.machine()})")
    
    major, minor, _ = sys.version_info[:3]
    if (major, minor) >= (3, 10):
        print(f"  * Runtime Check   : {GREEN}[PASS] Python version meets requirements (>= 3.10){RESET}\n")
        return True
    else:
        print(f"  * Runtime Check   : {RED}[FAIL] Python version is below 3.10{RESET}\n")
        return False


def check_project_directories():
    print(f"{BOLD}[2/4] Verifying Project Architecture Directory Layout...{RESET}")
    expected_dirs = [
        "data/raw",
        "data/processed",
        "data/embeddings",
        "notebooks",
        "src",
        "models",
        "app",
        "tests",
    ]
    all_exist = True
    for d in expected_dirs:
        if os.path.isdir(d):
            print(f"  [+] Directory '{d}'{' ' * (22 - len(d))}: {GREEN}[EXISTS]{RESET}")
        else:
            print(f"  [+] Directory '{d}'{' ' * (22 - len(d))}: {YELLOW}[CREATED]{RESET}")
            os.makedirs(d, exist_ok=True)
    print()
    return all_exist


def check_library_imports():
    print(f"{BOLD}[3/4] Checking Required Computer Vision & ML Libraries...{RESET}")
    packages = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("cv2", "OpenCV (cv2)"),
        ("PIL", "Pillow (PIL)"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("sklearn", "Scikit-Learn"),
        ("tensorflow", "TensorFlow"),
        ("keras", "Keras"),
        ("streamlit", "Streamlit"),
    ]

    all_passed = True
    missing_packages = []
    for module_name, display_name in packages:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "Installed")
            print(f"  [+] {display_name:<18} : {GREEN}{ver:<16}{RESET} {GREEN}[OK]{RESET}")
        except ImportError as e:
            print(f"  [-] {display_name:<18} : {RED}Not Installed ({e}){RESET}")
            all_passed = False
            missing_packages.append(display_name)

    if missing_packages:
        print(f"\n  {YELLOW}Note: Missing packages can be installed via 'pip install -r requirements.txt'{RESET}")

    print()
    return all_passed


def run_functional_smoke_tests():
    print(f"{BOLD}[4/4] Running Functional Computer Vision & ML Smoke Tests...{RESET}")
    tests_passed = True

    # Test 1: NumPy & Pandas
    try:
        import numpy as np
        import pandas as pd
        arr = np.random.rand(10, 5)
        df = pd.DataFrame(arr, columns=[f"feat_{i}" for i in range(5)])
        _ = df.mean().sum()
        print(f"  [+] Data Processing Test (NumPy & Pandas)          : {GREEN}[PASS]{RESET}")
    except Exception as e:
        print(f"  [-] Data Processing Test Failed                    : {RED}{e}{RESET}")
        tests_passed = False

    # Test 2: OpenCV & Pillow Image Manipulation
    try:
        import cv2
        import numpy as np
        from PIL import Image

        # Create a synthetic fashion garment canvas (224x224x3)
        img_bgr = np.zeros((224, 224, 3), dtype=np.uint8)
        # Draw a mock clothing shape (e.g. rectangular t-shirt torso)
        cv2.rectangle(img_bgr, (50, 40), (174, 200), (220, 180, 50), -1)
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_img = Image.fromarray(img_rgb)
        resized_pil = pil_img.resize((128, 128))
        assert resized_pil.size == (128, 128)
        print(f"  [+] Computer Vision Test (OpenCV & Pillow)         : {GREEN}[PASS]{RESET}")
    except Exception as e:
        print(f"  [-] Computer Vision Test Failed                    : {RED}{e}{RESET}")
        tests_passed = False

    # Test 3: TensorFlow & Keras Tensor Execution
    try:
        import tensorflow as tf
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        _ = tf.matmul(a, b)
        gpu_avail = len(tf.config.list_physical_devices('GPU')) > 0
        device_note = "GPU detected" if gpu_avail else "Running on CPU (Ready for inference & feature extraction)"
        print(f"  [+] Deep Learning Test (TensorFlow Matrix MatMul)  : {GREEN}[PASS]{RESET} ({device_note})")
    except Exception as e:
        print(f"  [-] Deep Learning Test Failed                      : {RED}{e}{RESET}")
        tests_passed = False

    # Test 4: Scikit-learn Cosine Similarity (Outfit Matching Logic)
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        # Simulate two 512-dim clothing feature embeddings
        vec_shirt = np.random.randn(1, 512)
        vec_jeans = np.random.randn(1, 512)
        sim_score = float(cosine_similarity(vec_shirt, vec_jeans)[0][0])
        print(f"  [+] Similarity Metric Test (Cosine Similarity)     : {GREEN}[PASS]{RESET} (Mock Sim: {sim_score:.4f})")
    except Exception as e:
        print(f"  [-] Similarity Metric Test Failed                  : {RED}{e}{RESET}")
        tests_passed = False

    print()
    return tests_passed


def main():
    print_banner()
    py_ok = check_python_environment()
    dirs_ok = check_project_directories()
    libs_ok = check_library_imports()
    smoke_ok = run_functional_smoke_tests() if libs_ok else False

    print(f"{CYAN}{'=' * 75}{RESET}")
    if py_ok and libs_ok and smoke_ok and dirs_ok:
        print(f"{BOLD}{GREEN}[SUCCESS] Module 2 Verification Completed Successfully!{RESET}")
        print(f"{GREEN}Your environment is 100% prepared for Module 3 (Fashion Dataset Collection & Understanding).{RESET}")
        return 0
    else:
        print(f"{BOLD}{YELLOW}[NOTICE] Some dependencies or tests need attention.{RESET}")
        print(f"Run 'pip install -r requirements.txt' to install missing dependencies.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
