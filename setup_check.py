import importlib
import pkg_resources
import sys
import re

# Pretty color output
class color:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

# Required packages and versions
required = {
    # Core PyTorch stack
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "torchaudio": "2.6.0",

    # Intel XPU extensions (optional)
    "intel-extension-for-pytorch": "2.6.10+xpu",
    "oneccl-bind-pt": "2.6.0+xpu",

    # Common ML deps
    "transformers": "4.49.0",
    "accelerate": "1.4.0",
    "bitsandbytes": "0.45.3",
    "peft": "0.15.1",
    "deepspeed": "0.16.4",
    "datasets": "3.3.2",
    "tiktoken": "0.9.0",
    "lm_eval": "0.4.8",
    "sentencepiece": None,
    "evaluate": None,
    "scipy": None,
    "tqdm": None,
    "regex": None,
    "sklearn": None,  # Changed from scikit-learn to sklearn
}

# XPU-only packages (will only be checked if XPU is available)
xpu_only_packages = {
    "oneccl": "2021.14.1",
    "oneccl-devel": "2021.14.1",
    "intel-cmplr-lib-rt": "2025.0.2",
    "intel-cmplr-lib-ur": "2025.0.2",
    "intel-cmplr-lic-rt": "2025.0.2",
    "intel-opencl-rt": "2025.0.4",
    "intel-openmp": "2025.0.4",
    "intel-pti": "0.10.0",
    "intel-sycl-rt": "2025.0.2",
}

def normalize_version(version_str):
    """Remove build suffixes like +xpu, +cu118, etc. for comparison"""
    if not version_str:
        return version_str
    return re.sub(r'\+.*$', '', version_str)

def check_xpu_available():
    """Check if Intel XPU is available"""
    try:
        import torch
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except:
        return False

def get_package_info(pkg_name):
    """Get package info, handling different package name variations"""
    # Common package name mappings
    name_mappings = {
        "sklearn": ["scikit-learn", "sklearn"],
        "oneccl-bind-pt": ["oneccl-bind-pt", "oneccl_bind_pt"],
        "intel-extension-for-pytorch": ["intel-extension-for-pytorch", "intel_extension_for_pytorch"],
    }

    # Import name mappings
    import_mappings = {
        "oneccl-bind-pt": "oneccl_bindings_for_pytorch",
        "intel-extension-for-pytorch": "intel_extension_for_pytorch",
        "sklearn": "sklearn",
    }

    # Try different package names
    possible_names = name_mappings.get(pkg_name, [pkg_name])

    for name in possible_names:
        try:
            version = pkg_resources.get_distribution(name).version
            return name, version
        except pkg_resources.DistributionNotFound:
            continue

    return None, None

def get_import_name(pkg_name):
    """Get the correct import name for a package"""
    import_mappings = {
        "oneccl-bind-pt": "oneccl_bindings_for_pytorch",
        "intel-extension-for-pytorch": "intel_extension_for_pytorch",
        "sklearn": "sklearn",
    }

    return import_mappings.get(pkg_name, pkg_name.replace("-", "_"))

print("\n🔍 Checking Python environment...\n")

ok_all = True
is_xpu_system = check_xpu_available()

# Check main packages
for pkg, expected_version in required.items():
    try:
        # Get the correct package name and version
        actual_pkg_name, installed_version = get_package_info(pkg)

        if actual_pkg_name is None:
            print(f"{color.RED}❌ {pkg}: not installed{color.RESET}")
            ok_all = False
            continue

        # Get import name and try to import
        import_name = get_import_name(pkg)
        importlib.import_module(import_name)

        if expected_version:
            # Normalize versions for comparison (remove +xpu, +cu118, etc.)
            normalized_installed = normalize_version(installed_version)
            normalized_expected = normalize_version(expected_version)

            if normalized_installed != normalized_expected:
                print(f"{color.YELLOW}⚠️   {pkg}: installed {installed_version}, expected {expected_version}{color.RESET}")
            else:
                print(f"{color.GREEN}✅ {pkg}: {installed_version}{color.RESET}")
        else:
            print(f"{color.GREEN}✅ {pkg}: {installed_version}{color.RESET}")

    except ImportError as e:
        # Special handling for wandb numpy error
        if pkg == "wandb" and "np.float_" in str(e):
            print(f"{color.YELLOW}⚠️   {pkg}: installed but has NumPy 2.0 compatibility issue{color.RESET}")
            print(f"    {color.YELLOW}Suggestion: pip install --upgrade wandb numpy{color.RESET}")
        else:
            print(f"{color.RED}❌ {pkg}: import failed - {str(e)[:100]}...{color.RESET}")
            ok_all = False
    except Exception as e:
        print(f"{color.RED}❌ {pkg}: error ({e}){color.RESET}")
        ok_all = False

# Check XPU-specific packages only if XPU is available
if is_xpu_system:
    print(f"\n{color.BLUE}🔧 Checking XPU-specific packages...{color.RESET}")

    for pkg, expected_version in xpu_only_packages.items():
        try:
            installed_version = pkg_resources.get_distribution(pkg).version

            if expected_version:
                normalized_installed = normalize_version(installed_version)
                normalized_expected = normalize_version(expected_version)

                if normalized_installed != normalized_expected:
                    print(f"{color.YELLOW}⚠️   {pkg}: installed {installed_version}, expected {expected_version}{color.RESET}")
                else:
                    print(f"{color.GREEN}✅ {pkg}: {installed_version}{color.RESET}")
            else:
                print(f"{color.GREEN}✅ {pkg}: {installed_version}{color.RESET}")

        except pkg_resources.DistributionNotFound:
            print(f"{color.YELLOW}⚠️   {pkg}: not found (XPU package){color.RESET}")
        except Exception as e:
            print(f"{color.YELLOW}⚠️   {pkg}: error ({e}){color.RESET}")
else:
    print(f"\n{color.BLUE}ℹ️  XPU not detected - skipping XPU-specific packages{color.RESET}")

# PyTorch device check
try:
    import torch
    print(f"\n{color.BLUE}🖥️  Device Detection:{color.RESET}")

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        print(f"{color.GREEN}✅ CUDA GPU detected: {device_name} (CUDA {cuda_version}){color.RESET}")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        print(f"{color.GREEN}✅ Intel XPU detected{color.RESET}")
        try:
            device_count = torch.xpu.device_count()
            print(f"   XPU devices available: {device_count}")
        except:
            pass
    else:
        print(f"{color.YELLOW}⚠️   No GPU/XPU detected — running on CPU{color.RESET}")

except Exception as e:
    print(f"{color.RED}❌ torch device check failed: {e}{color.RESET}")

# Summary
print(f"\n{'='*50}")
if ok_all:
    print(f"{color.GREEN}✅ Environment check complete - all core packages OK!{color.RESET}")
else:
    print(f"{color.YELLOW}⚠️  Some packages are missing or have issues.{color.RESET}")

if is_xpu_system:
    print(f"{color.BLUE}ℹ️  Intel XPU system detected{color.RESET}")

print(f"{'='*50}\n")
