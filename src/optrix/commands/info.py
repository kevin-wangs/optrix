"""Info command implementation."""
def run_info(args):
    from optrix.core import detect_devices
    for d in detect_devices():
        print(d)
