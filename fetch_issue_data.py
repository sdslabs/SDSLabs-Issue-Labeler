import os
import aiohttp
from gidgethub.aiohttp import GitHubAPI
import asyncio
import gspread
import re
from oauth2client.service_account import ServiceAccountCredentials

scope = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file'
]
defined_labels = ['enhancement', 'bug', 'help wanted', 'question', 'wontfix', 'feature', 'good first issue']

file_name = 'client_key.json'
creds = ServiceAccountCredentials.from_json_keyfile_name(file_name, scope)
client = gspread.authorize(creds)
sheet = client.open('Dataset Github').sheet1

async def main():
    async with aiohttp.ClientSession() as session:
        gh = GitHubAPI(session, "mhk19", oauth_token=os.getenv("GH_AUTH"))
        count = 1
        i = 0
        while(True):
            array = (await gh.getitem('/repos/mozilla/fxa/issues?page='+str(i)+'&per_page=100'))
            print(i)
            if len(array)==0:
                break
            start = count+1
            rows = []
            for issue in array:
                x = {
                    "enhancement": 0,
                    "bug": 0,
                    "help wanted": 0,
                    "question": 0,
                    "wontfix": 0,
                    "feature": 0,
                    "good first issue": 0
                }
                for label in issue['labels']:
                    k = label['name']
                    print(k)
                    m = (re.sub(r"[^a-zA-Z0-9]+", '', k))
                    if m in defined_labels:
                        x[m] = 1
                title = issue['title']+' '+issue['body']
                count = count + 1
                rows.append([count, title, x["enhancement"], x["bug"], x["help wanted"], x["question"], x["wontfix"], x["feature"], x["good first issue"]])
            sheet.insert_rows(rows, start)
            i = i+1
        print(count)


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
