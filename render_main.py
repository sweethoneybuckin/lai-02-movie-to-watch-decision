import os
import threading
from aiohttp import web
import moviedecision

# Simple HTTP handler for Render keep-alive ping
async def handle_ping(request):
    return web.Response(text="Bot Discord Movie Decision Status: ONLINE 24/7 🟢")

def start_web_server():
    app = web.Application()
    app.router.add_get('/', handle_ping)
    port = int(os.getenv("PORT", 10000))
    web.run_app(app, host='0.0.0.0', port=port, print=None)

if __name__ == "__main__":
    # Start web server in background thread for Render
    threading.Thread(target=start_web_server, daemon=True).start()
    
    # Run Discord bot main loop
    moviedecision.main()
