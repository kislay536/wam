"""
Policy Broker — middleware connecting N policy servers with M environment clients.

Message Protocol
--------
Frontend (ROUTER, facing client DEALER)
  recv : [client_id, data_pkl]
  send : [client_id, result_pkl]

Backend (ROUTER, facing server DEALER)
  recv READY  : [server_id, b"READY"]
  recv RESULT : [server_id, b"RESULT", client_id, result_pkl]
  send WORK   : [server_id, b"WORK",   client_id, data_pkl]

Dispatch Logic
--------
Whenever there are both idle servers and pending requests in the queue, dispatch 1:1 immediately without batching.
"""

import os
import sys
import pickle
import logging
from collections import deque
from dataclasses import dataclass

import zmq
import tyro

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Args:
    frontend_port: int = 10086
    """Port for client connections (consistent with original server_port)"""
    backend_port: int = 10087
    """Port for server connections"""
    host: str = "*"


def main(args: Args):
    context = zmq.Context()

    frontend = context.socket(zmq.ROUTER)
    frontend.bind(f"tcp://{args.host}:{args.frontend_port}")

    backend = context.socket(zmq.ROUTER)
    backend.bind(f"tcp://{args.host}:{args.backend_port}")

    logging.info(f"Broker started  " f"frontend=:{args.frontend_port}  backend=:{args.backend_port}")

    available_servers: deque[bytes] = deque()
    pending_requests: deque[tuple[bytes, bytes]] = deque()  # (client_id, data_pkl)

    poller = zmq.Poller()
    poller.register(frontend, zmq.POLLIN)
    poller.register(backend, zmq.POLLIN)

    while True:
        socks = dict(poller.poll())

        # ── Handle backend messages (from servers) ─────────────────────────────
        if backend in socks:
            frames = backend.recv_multipart()
            server_id, msg_type = frames[0], frames[1]

            if msg_type == b"READY":
                available_servers.append(server_id)
                logging.info(f"Server {server_id.hex()[:8]} READY  " f"available={len(available_servers)}")

            elif msg_type == b"RESULT":
                # frames: [server_id, b"RESULT", client_id, result_pkl]
                client_id, result_pkl = frames[2], frames[3]
                frontend.send_multipart([client_id, result_pkl])
                logging.info(f"Forwarded result to client {client_id.hex()[:8]}")

        # ── Handle frontend messages (from clients) ────────────────────────────
        if frontend in socks:
            frames = frontend.recv_multipart()
            client_id, data_pkl = frames[0], frames[1]
            pending_requests.append((client_id, data_pkl))

        # ── Immediate dispatch: send whenever there are requests and idle servers
        while available_servers and pending_requests:
            server_id = available_servers.popleft()
            client_id, data_pkl = pending_requests.popleft()
            backend.send_multipart([server_id, b"WORK", client_id, data_pkl])
            logging.info(
                f"Dispatched → server {server_id.hex()[:8]}  "
                f"pending={len(pending_requests)}  available={len(available_servers)}"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(tyro.cli(Args))
