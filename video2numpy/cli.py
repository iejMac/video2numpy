"""cli entry point"""

import fire

from video2numpy import video2numpy


def main():
    """Main entry point"""
    fire.Fire(video2numpy)


if __name__ == "__main__":
    main()
