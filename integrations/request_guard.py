"""Pre-parser body/concurrency bounds and optional server-side bearer authentication."""
import asyncio
import hmac
import os
from threading import BoundedSemaphore
import time
from starlette.responses import JSONResponse


class RequestGuard:
    def __init__(self, app):
        self.app = app
        self.slots = BoundedSemaphore(4)
        self.body_limit = 6 * 1024 * 1024

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or not scope['path'].startswith('/api/') or scope['method'] == 'OPTIONS':
            return await self.app(scope, receive, send)
        headers = dict(scope['headers'])
        token = os.getenv('API_AUTH_TOKEN', '').strip()
        if token:
            supplied = headers.get(b'authorization', b'')
            if not hmac.compare_digest(supplied, ('Bearer ' + token).encode()):
                return await JSONResponse({'detail': 'A valid access token is required.'}, 401)(scope, receive, send)
        if not self.slots.acquire(blocking=False):
            return await JSONResponse({'detail': 'The server is busy. Retry shortly.'}, 429, headers={'Retry-After': '5'})(scope, receive, send)
        try:
            content_length = headers.get(b'content-length')
            if content_length:
                try:
                    oversized = int(content_length) > self.body_limit
                except ValueError:
                    return await JSONResponse({'detail': 'Invalid content length.'}, 400)(scope, receive, send)
                if oversized:
                    return await JSONResponse({'detail': 'Request body exceeds 6 MB.'}, 413)(scope, receive, send)
            body, total = [], 0
            deadline = time.monotonic() + 30
            while True:
                try:
                    message = await asyncio.wait_for(receive(), timeout=max(0.001, deadline-time.monotonic()))
                except asyncio.TimeoutError:
                    return await JSONResponse({'detail': 'Request upload timed out.'}, 408)(scope, receive, send)
                if message['type'] == 'http.disconnect':
                    return
                chunk = message.get('body', b''); total += len(chunk)
                if total > self.body_limit:
                    return await JSONResponse({'detail': 'Request body exceeds 6 MB.'}, 413)(scope, receive, send)
                body.append(chunk)
                if not message.get('more_body', False):
                    break
            delivered = False
            async def bounded_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {'type': 'http.request', 'body': b''.join(body), 'more_body': False}
                return await receive()
            async def private_send(message):
                if message['type'] == 'http.response.start':
                    message['headers'] = [*message.get('headers', []), (b'cache-control', b'no-store'), (b'x-content-type-options', b'nosniff')]
                await send(message)
            # Keep the slot until real processing finishes, including client cancellation.
            task = asyncio.create_task(self.app(scope, bounded_receive, private_send))
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise
        finally:
            self.slots.release()
