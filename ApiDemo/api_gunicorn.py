import sys
import json
import logging
from os import getenv
from aiohttp import web
from json.decoder import JSONDecodeError


PORT = getenv("PORT", "5000")
LOG_LEVEL = getenv("LOG_LEVEL", "DEBUG")
LOGGER = logging.getLogger("main")
levels = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}
LOGGER.setLevel(levels[LOG_LEVEL])

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(levels[LOG_LEVEL])
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
LOGGER.addHandler(ch)


async def healthcheck(request):
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return web.Response(text=json.dumps("Healthy API"), headers=headers, status=200)


async def test_function(request):
    content_type = request.headers.get("Content-Type")
    try:
        input_data = await request.json()
    except JSONDecodeError as err:
        LOGGER.error(f"Invalid JSON error: {err}")
        return web.Response(text=json.dumps(f"Invalid JSON error: {err}"), status=400)

    if content_type != "application/json":
        LOGGER.error("Content-Type header must be 'application/json':")
        return web.Response(
            text=json.dumps("Content-Type header must have value of application/json"),
            status=400)
    try:
        return web.Response(
            body=json.dumps({"repo": "api test"}),
            headers=dict({"Content-Type": "application/json"}),
            status=200)
    except Exception as err:
        LOGGER.error(f"Internal server error: {err}")
        return web.Response(text=json.dumps("Internal server error"), status=500)


"""Define app and API endpoints"""
app = web.Application()
app.router.add_get("/api_example/healthcheck", healthcheck)
app.router.add_post("/api_example/test", test_function)


if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=(int(PORT)))
