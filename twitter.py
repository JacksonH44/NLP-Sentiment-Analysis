import tweepy
from api_secrets import *

import tweepy.auth

# TODO: Authenticate Twitter app so I can live stream tweets into the machine
class TwitterStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        print(status)

    def on_error(self, status_code):
        if status_code == 403:
            print("Access has been refused")
        return False


if __name__ == '__main__':

    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True,
                     retry_count=10, retry_delay=5, retry_errors=5)

    # api.update_status()

    stream_listener = TwitterStreamListener()
    my_stream = tweepy.Stream(auth=api.auth, listener=stream_listener)

    my_stream.filter(track=["jackson"], is_async=True)
