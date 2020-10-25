import os
from os import error
import aiohttp
from aiohttp import web
from gidgethub import routing, sansio
from gidgethub import aiohttp as gh_aiohttp
from github_bot import get_labels
import yaml

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
yamlFilePath = CUR_DIR + "/config.yml"
try:
    confFile = open(yamlFilePath, "r")
except FileNotFoundError:
    print("The required config file " + yamlFilePath + " does not exist")
    exit(1)
ENV_VARS = yaml.safe_load(confFile)

router = routing.Router()

routes = web.RouteTableDef()

@router.register("issues", action="opened")
async def issue_opened_event(event, gh, *args, **kwargs):
    issue = event.data["issue"]
    url = issue["url"]+"/labels"
    title = issue["title"]
    body = "this is hardcoded body"
    labels = await get_labels(title, body)
    filtered_labels = []

    for label_item in labels:
        print(label_item['name'] ,label_item['probability'])
        if(label_item['probability']>0.6):
            filtered_labels.append(label_item['name'])

    if(len(filtered_labels) < 2):
        sorted_labels = sorted(labels, key = lambda i: i['probability'],reverse=True)
        filtered_labels= [sorted_labels[0]['name'],sorted_labels[1]['name']]

    await gh.post(url, data={"labels": filtered_labels})

@routes.post("/")
async def main(request):
    body = await request.read()
    secret = ENV_VARS['secret_key']
    oauth_token = ENV_VARS['oauth_token']
    username = ENV_VARS['username']
    event = sansio.Event.from_http(request.headers, body, secret=secret)
    async with aiohttp.ClientSession() as session:
        gh = gh_aiohttp.GitHubAPI(session, username, oauth_token=oauth_token)
        await router.dispatch(event, gh)
    return web.Response(status=200)

if __name__ == "__main__":
    app = web.Application()
    app.add_routes(routes)
    port = ENV_VARS['port']
    if port is not None:
        port = int(port)
    else:
        print("Port not found")
    
    web.run_app(app, port=port)
