import asyncio
import json
import socket
import time
from typing import List, Dict, Callable, Tuple, Coroutine
from ..discovery import Discovery
from ..peer_handle import PeerHandle
from .grpc_peer_handle import GRPCPeerHandle
from exo.topology.device_capabilities import DeviceCapabilities, device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from exo import DEBUG_DISCOVERY

# Load the valid IP addresses from the file
def load_valid_ips(filepath):
    with open(filepath, 'r') as f:
        return {line.strip() for line in f.readlines()}

valid_supernode_ips_file_path = "/home/ubuntu/python_inference_layer_server/valid_supernode_list.txt"
valid_ips = load_valid_ips(valid_supernode_ips_file_path)

class ListenProtocol(asyncio.DatagramProtocol):
    def __init__(self, on_message: Callable[[bytes, Tuple[str, int]], Coroutine]):
        super().__init__()
        self.on_message = on_message
        self.loop = asyncio.get_event_loop()

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        asyncio.create_task(self.on_message(data, addr))


class GRPCDiscovery(Discovery):
    def __init__(
        self,
        node_id: str,
        node_port: int,
        listen_port: int,
        broadcast_port: int = None,
        broadcast_interval: int = 1,
        device_capabilities: DeviceCapabilities = UNKNOWN_DEVICE_CAPABILITIES,
        discovery_timeout: int = 30,
    ):
        self.node_id = node_id
        self.node_port = node_port
        self.device_capabilities = device_capabilities
        self.listen_port = listen_port
        self.broadcast_port = broadcast_port if broadcast_port is not None else listen_port
        self.broadcast_interval = broadcast_interval
        self.known_peers: Dict[str, Tuple[GRPCPeerHandle, float, float]] = {}
        self.broadcast_task = None
        self.listen_task = None
        self.cleanup_task = None
        self.discovery_timeout = discovery_timeout

    async def start(self):
        self.device_capabilities = device_capabilities()
        self.broadcast_task = asyncio.create_task(self.task_broadcast_presence())
        self.listen_task = asyncio.create_task(self.task_listen_for_peers())
        self.cleanup_task = asyncio.create_task(self.task_cleanup_peers())

    async def stop(self):
        if self.broadcast_task:
            self.broadcast_task.cancel()
        if self.listen_task:
            self.listen_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.broadcast_task or self.listen_task or self.cleanup_task:
            await asyncio.gather(self.broadcast_task, self.listen_task, self.cleanup_task, return_exceptions=True)

    async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
        if DEBUG_DISCOVERY >= 2:
            print("Starting peer discovery process...")

        while len(self.known_peers) < wait_for_peers:
            if DEBUG_DISCOVERY >= 2:
                print(f"Waiting for more peers. Current: {len(self.known_peers)}, Required: {wait_for_peers}")
            await asyncio.sleep(1)

        # Filter known_peers based on valid_ips
        valid_peers = [
            peer_handle for peer_handle, _, _ in self.known_peers.values()
            if peer_handle.address.split(':')[0] in valid_ips
        ]

        if DEBUG_DISCOVERY >= 2:
            print(f"Discovered {len(valid_peers)} valid peers out of {len(self.known_peers)} total peers")

        return valid_peers

    async def task_broadcast_presence(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind(('', 0))

        message = json.dumps(
            {
                "type": "discovery",
                "node_id": self.node_id,
                "grpc_port": self.node_port,
                "device_capabilities": self.device_capabilities.to_dict(),
            }
        ).encode("utf-8")

        while True:
            try:
                if DEBUG_DISCOVERY >= 3:
                    print(f"Broadcasting presence to valid IPs: {message}")
                for ip in valid_ips:
                    sock.sendto(message, (ip, self.broadcast_port))
                await asyncio.sleep(self.broadcast_interval)
            except Exception as e:
                print(f"Error in broadcast presence: {e}")
                import traceback
                print(traceback.format_exc())

    async def on_listen_message(self, data, addr):
        if not data:
            return

        try:
            message = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError:
            if DEBUG_DISCOVERY >= 2:
                print(f"Received invalid JSON data from {addr}")
            return

        if DEBUG_DISCOVERY >= 2:
            print(f"Received message from peer {addr}: {message}")

        if message["type"] == "discovery" and message["node_id"] != self.node_id:
            peer_id = message["node_id"]
            peer_host = addr[0]  # This is the actual IP address of the peer
            peer_port = message["grpc_port"]
            device_capabilities = DeviceCapabilities(**message["device_capabilities"])

            if peer_host in valid_ips:
                if peer_id not in self.known_peers:
                    print(f"Discovered peer: {peer_id} at {peer_host}:{peer_port}")
                    self.known_peers[peer_id] = (
                        GRPCPeerHandle(peer_id, f"{peer_host}:{peer_port}", device_capabilities),
                        time.time(),
                        time.time(),
                    )
                else:
                    # Update the existing peer's last seen time
                    existing_peer, connected_at, _ = self.known_peers[peer_id]
                    self.known_peers[peer_id] = (existing_peer, connected_at, time.time())
            else:
                if DEBUG_DISCOVERY >= 1:
                    print(f"Received message from {peer_id} at {peer_host}, but IP is not in valid list. Message: {message}")

    async def task_listen_for_peers(self):
        await asyncio.get_event_loop().create_datagram_endpoint(
            lambda: ListenProtocol(self.on_listen_message),
            local_addr=("0.0.0.0", self.listen_port)
        )
        if DEBUG_DISCOVERY >= 2:
            print("Started listen task")

    async def task_cleanup_peers(self):
        while True:
            try:
                current_time = time.time()
                peers_to_remove = [
                    peer_handle.id()
                    for peer_handle, connected_at, last_seen in self.known_peers.values()
                    if (not await peer_handle.is_connected() and current_time - connected_at > self.discovery_timeout) or current_time - last_seen > self.discovery_timeout
                ]
                for peer_id in peers_to_remove:
                    if peer_id in self.known_peers:
                        del self.known_peers[peer_id]
                    if DEBUG_DISCOVERY >= 2:
                        print(f"Removed peer {peer_id} due to inactivity.")
                await asyncio.sleep(self.broadcast_interval)
            except Exception as e:
                print(f"Error in cleanup peers: {e}")
                import traceback
                print(traceback.format_exc())
