"""
The wire between a dog and the central computer: one TCP connection per dog (the dog listens,
the central computer connects), messages both ways. A message is

    [u32 header length][header: JSON][u32 body length][body]        (lengths big-endian)

and the header's "ch" names its channel. No ROS here, and Python 3.8 (the dog's ROS 2 Foxy), so
both sides use this file whatever their ROS versions: dog_bridge.py on the dog, dog_link.py here.

  dog -> central   hello  {robot, nav_ready, mission}                           1 Hz
                          mission: null, or {id, task, hint, targets} the robot asks for
                   map    {frame, resolution, width, height, origin: [x, y, z, qx, qy, qz, qw]}
                          body: zlib of the int8 cells, row-major                each new map, <= 1 Hz
                   pose   {tf: [x, y, z, qx, qy, qz, qw]}  the base in the map frame      10 Hz
                   dets   {items: [[class, score], ...]}                       as detected
                   nav    {id, status, code}  status: accepted, rejected, succeeded, aborted, canceled
  central -> dog   goal   {id, x, y, yaw}  in the dog's map frame
                   cancel {id}

Each side stamps what it receives with its own clock: the two clocks need not agree.
"""

import json
import struct

PORT = 9100
MAX_LEN = 64 * 1024 * 1024           # a sanity bound on one part


def pack(header, body=b""):
    h = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack(">I", len(h)) + h + struct.pack(">I", len(body)) + body


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        part = sock.recv(min(n - len(buf), 1 << 20))
        if not part:
            raise ConnectionError("connection closed")
        buf += part
    return bytes(buf)


def recv(sock):
    """(header dict, body bytes) of the next message; ConnectionError when the peer is gone."""
    n = struct.unpack(">I", _recv_exact(sock, 4))[0]
    if n > MAX_LEN:
        raise ConnectionError("header of %d bytes" % n)
    header = json.loads(_recv_exact(sock, n).decode())
    n = struct.unpack(">I", _recv_exact(sock, 4))[0]
    if n > MAX_LEN:
        raise ConnectionError("body of %d bytes" % n)
    return header, _recv_exact(sock, n)
