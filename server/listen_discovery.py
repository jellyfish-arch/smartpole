"""Pretend to be the ESP32 at boot: listen for the server's UDP broadcast.

With the server running, in a second terminal:
    python -m server.listen_discovery

If this prints "server found", the broadcaster works. Run it on a second
laptop or phone-connected device on the same hotspot to test the real path.
"""

import socket

from config import settings


def main() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", settings.DISCOVERY_PORT))
    sock.settimeout(10)
    print(f"Listening on UDP {settings.DISCOVERY_PORT} for 10 s...")
    try:
        data, (ip, _) = sock.recvfrom(64)
        print(f"server found: '{data.decode()}' from {ip}")
    except socket.timeout:
        print("nothing heard. Is the server running?")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
