"""UDP discovery broadcaster.

Every couple of seconds the server shouts "SMARTPOLE_SERVER 8000" to every
device on the local network. The ESP32 listens for this at boot and takes
the sender's address as the server address. This way the firmware does not
depend on the laptop keeping the same IP on the phone hotspot.

Why send to each network's own broadcast address (e.g. 192.168.43.255)
instead of just 255.255.255.255: on Windows, a 255.255.255.255 broadcast
leaves through only one network adapter, which may not be the hotspot one.
"""

import socket
import threading

import psutil

from config import settings


def broadcast_addresses() -> list[str]:
    """Broadcast address of every active IPv4 network, e.g. '192.168.43.255'."""
    addresses = set()
    stats = psutil.net_if_stats()
    for name, addrs in psutil.net_if_addrs().items():
        if name in stats and not stats[name].isup:
            continue
        for a in addrs:
            if a.family != socket.AF_INET or a.address.startswith("127."):
                continue
            if a.broadcast:
                addresses.add(a.broadcast)
            elif a.netmask:
                # Compute it ourselves: host bits all set to 1.
                ip = int.from_bytes(socket.inet_aton(a.address), "big")
                mask = int.from_bytes(socket.inet_aton(a.netmask), "big")
                addresses.add(socket.inet_ntoa(((ip | ~mask) & 0xFFFFFFFF).to_bytes(4, "big")))
    return sorted(addresses)


def run_broadcaster(stop_event: threading.Event) -> None:
    message = f"SMARTPOLE_SERVER {settings.SERVER_PORT}".encode()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    print(f"[discovery] broadcasting on UDP {settings.DISCOVERY_PORT} to {broadcast_addresses()}")
    try:
        while not stop_event.is_set():
            # Re-read the addresses every time: joining a hotspot after the
            # server started adds a new network.
            for addr in broadcast_addresses():
                try:
                    sock.sendto(message, (addr, settings.DISCOVERY_PORT))
                except OSError:
                    pass   # adapter went away; try again next round
            stop_event.wait(settings.DISCOVERY_INTERVAL_S)
    finally:
        sock.close()


def start_in_background() -> threading.Event:
    """Start the broadcaster thread. Set the returned event to stop it."""
    stop_event = threading.Event()
    threading.Thread(target=run_broadcaster, args=(stop_event,), daemon=True).start()
    return stop_event
