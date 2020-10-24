import os
import aiohttp
from aiohttp import web
from aiohttp.client import request
from gidgethub import routing, sansio
from gidgethub import aiohttp as gh_aiohttp

router = routing.Router()

routes = web.RouteTableDef()

@router.register("issues", action="opened")
async def issue_opened_event(event, gh, *args, **kwargs):
    url = event.data["issue"]["url"]+"/labels"
    label = "test"
    await gh.post(url, data={"labels":[label]})

@routes.post("/")
async def main(request):
    body = await request.read()
    secret = os.environ.get("GH_SECRET")
    oauth_token = os.environ.get("GH_AUTH")
    event = sansio.Event.from_http(request.headers, body, secret=secret)
    async with aiohttp.ClientSession() as session:
        gh = gh_aiohttp.GitHubAPI(session, "mhk19", oauth_token=oauth_token)
        await router.dispatch(event, gh)
    return web.Response(status=200)

if __name__ == "__main__":
    app = web.Application()
    app.add_routes(routes)
    port = os.environ.get("PORT")
    if port is not None:
        port = int(port)
    
    web.run_app(app, port=port)
