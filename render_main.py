import os
import threading
from aiohttp import web
import moviedecision

# Handler HTTP untuk Health Check SnapDeploy
async def handle_ping(request):
    return web.Response(text="Bot Discord Movie Decision Status: ONLINE 24/7 🟢")

def start_bot():
    moviedecision.main()

if __name__ == "__main__":
    # 1. Jalankan Bot Discord di Background Thread
    threading.Thread(target=start_bot, daemon=True).start()
    
    # 2. Jalankan Web Server Aiohttp di Main Thread (Main Interpreter)
    app = web.Application()
    app.router.add_get('/', handle_ping)
    port = int(os.getenv("PORT", 10000))
    web.run_app(app, host='0.0.0.0', port=port, print=None)