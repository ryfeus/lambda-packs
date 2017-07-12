import requests
from lxml import html

def handler(event, context):
    url = "https://twitter.com/realDonaldTrump"
    response = requests.request("GET", url)
    tree = html.fromstring(response.content)
    vecTweets = tree.xpath('//div[@class="js-tweet-text-container"]//p')	
    return vecTweets[0].text_content()
