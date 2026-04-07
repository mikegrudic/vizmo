"""DataFlyer entry point."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="DataFlyer - Real-time mesh-free data explorer")
    parser.add_argument("snapshot", help="Path to HDF5 snapshot file")
    parser.add_argument("--width", type=int, default=1920, help="Window width")
    parser.add_argument("--height", type=int, default=1080, help="Window height")
    parser.add_argument("--fov", type=float, default=90.0,
                        help="Field of view in degrees")
    parser.add_argument("--screenshot", type=str, default=None, metavar="OUT",
                        help="Render one frame to OUT (PNG) after GPU init "
                             "+ auto-range complete, then exit")
    parser.add_argument("--fullscreen", action="store_true",
                        help="Run in fullscreen mode at specified resolution")
    parser.add_argument("--profile", type=str, default=None, metavar="OUT",
                        help="Profile the whole run with cProfile and dump "
                             "stats to OUT (.pstats). View with snakeviz.")
    args = parser.parse_args()

    from .wgpu_app import run_wgpu_app
    if args.profile:
        import cProfile
        import pstats
        pr = cProfile.Profile()
        pr.enable()
        try:
            run_wgpu_app(args.snapshot, width=args.width, height=args.height,
                         fov=args.fov, fullscreen=args.fullscreen,
                         screenshot=args.screenshot)
        finally:
            pr.disable()
            pr.dump_stats(args.profile)
            stats = pstats.Stats(pr).sort_stats("cumulative")
            print("\n=== top 40 by cumulative time ===")
            stats.print_stats(40)
            print(f"\nFull profile written to {args.profile}")
            print(f"View with: snakeviz {args.profile}")
    else:
        run_wgpu_app(args.snapshot, width=args.width, height=args.height,
                     fov=args.fov, fullscreen=args.fullscreen,
                     screenshot=args.screenshot)


if __name__ == "__main__":
    main()
