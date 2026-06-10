"""Optrix CLI entry point."""
import argparse
import sys

def main(argv=None):
    parser = argparse.ArgumentParser(prog="optrix", description="Optrix compute toolkit")
    sub = parser.add_subparsers(dest="command")

    info_p = sub.add_parser("info", help="Show device info")
    info_p.add_argument("--device", type=int, default=0)

    bench_p = sub.add_parser("bench", help="Run benchmarks")
    bench_p.add_argument("--size", type=int, default=1024)
    bench_p.add_argument("--iterations", type=int, default=100)

    sub.add_parser("kernels", help="List registered kernels")

    args = parser.parse_args(argv)

    if args.command == "info":
        from optrix.core import detect_devices
        devices = detect_devices()
        dev = devices[min(args.device, len(devices) - 1)]
        print(f"Device {dev.device_id}: {dev.name}")
        print(f"  Architecture: {dev.arch}")
        print(f"  Memory: {dev.memory_gb:.1f} GB")
        print(f"  Compute Units: {dev.compute_units}")
        print(f"  Wavefront: {dev.wavefront_size}")
    elif args.command == "bench":
        print(f"Running benchmarks (size={args.size}, iter={args.iterations})...")
        from examples.benchmark import bench
        bench("test", lambda: None, args.iterations)
    elif args.command == "kernels":
        from optrix.dispatch import list_kernels
        for k in sorted(list_kernels()):
            print(f"  {k}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
