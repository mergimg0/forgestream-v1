"""WebSocket live updates for the dashboard."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

ws_router = APIRouter()

# Connected clients
_clients: list[WebSocket] = []


@ws_router.websocket("/ws/live")
async def live_updates(websocket: WebSocket) -> None:
    """Stream new events to the dashboard in real-time."""
    await websocket.accept()
    _clients.append(websocket)
    try:
        while True:
            # Keep connection alive, events are pushed via broadcast()
            await websocket.receive_text()
    except WebSocketDisconnect:
        _clients.remove(websocket)


async def broadcast(event_data: dict) -> None:
    """Broadcast an event to all connected dashboard clients."""
    disconnected = []
    for client in _clients:
        try:
            await client.send_json(event_data)
        except Exception:
            disconnected.append(client)
    for client in disconnected:
        _clients.remove(client)
