import requests
import os
import json
import urllib

class Tweet:
    def __init__(self, raw_tweet_dict):
        self.id = raw_tweet_dict['id']
        self.text = raw_tweet_dict['text']
        self.raw = raw_tweet_dict

class APIConfig:
    def __init__(self, conf_json_path='./local_configs/api_keys.json', lang=None):
        self.conf_json_path = conf_json_path
        with open(self.conf_json_path) as f:
            data = json.load(f)
        self.bearer = data['bearer']
        self.tweet_fields = ['text']
        self.options = {'max_results': 10,
                        'expansions': 'referenced_tweets.id'}

class APIManager:
    def __init__(self, config: APIConfig):
        self.config = config
        self.found_tweets = []

    def query(self, query: str, pages=1, original=True):
        """A simple query which gives back Tweets"""
        query = urllib.parse.quote(query)
        result_list = []
        headers = self._get_headers()
        options = self._stringify_options(self.config.options)
        tweet_fields = 'tweet.fields='+','.join(self.config.tweet_fields)
        
        next_token = None
        for page in range(pages):
            if not next_token:
                url = ('https://api.twitter.com/2/tweets/search/'
                        f'recent?query={query}&{options}&{tweet_fields}')
                json_response = self._connect_to_endpoint(url, headers)
            else:
                url = ('https://api.twitter.com/2/tweets/search/'
                        f'recent?query={query}&next_token={next_token}&{options}&{tweet_fields}')
                json_response = self._connect_to_endpoint(url, headers)
            result_list.append(json_response)
            if 'next_token' in json_response['meta'].keys():
                next_token = json_response['meta']['next_token']
            else:
                break
        tweets = self._extract_tweets(result_list)
        self.found_tweets = [Tweet(t) for t in tweets]
        # print(url)
        if not original:
            return self.found_tweets
        else:
            original_tweets = []
            for t in self.found_tweets:
                if self._is_original(t):
                    original_tweets.append(t)
                else:
                    original_t = self.get_tweet(self._get_original_tweet_id(t))
                    original_tweets.append(original_t)
            return original_tweets

    def get_tweet(self, id):
        tweet_fields = "tweet.fields=lang,author_id"
        # Tweet fields are adjustable.
        # Options include:
        # attachments, author_id, context_annotations,
        # conversation_id, created_at, entities, geo, id,
        # in_reply_to_user_id, lang, non_public_metrics, organic_metrics,
        # possibly_sensitive, promoted_metrics, public_metrics, referenced_tweets,
        # source, text, and withheld
        ids = f"ids={id}"
        # You can adjust ids to include a single Tweets.
        # Or you can add to up to 100 comma-separated IDs
        url = "https://api.twitter.com/2/tweets?{}&{}".format(ids, tweet_fields)
        headers = self._get_headers()
        json_response = self._connect_to_endpoint(url, headers)
        
        return Tweet(self._extract_tweets([json_response])[0])

    def _is_original(self, tweet):
        if 'referenced_tweets' in tweet.raw.keys():
            return False
        else:
            return True

    def _get_headers(self):
        return {"Authorization": "Bearer {}".format(self.config.bearer)}

    def _extract_tweets(self, query_result_list):
        tweets = []
        for page in query_result_list:
            tweets += page['data']
        return tweets

    def _stringify_options(self, options:{}) -> str:
        options_list = []
        for k in options.keys():
            options_list.append(f'{k}={options[k]}')
        return '&'.join(options_list)

    def _connect_to_endpoint(self, url, headers):
        # print(url)
        # print('--')
        response = requests.request("GET", url, headers=headers)
        json_response = None
        if response.status_code == 200:
            json_response = response.json()
            # response_data_len = len(json_response['data'])
            # print(f'got {response_data_len} tweets.')
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        return json_response

    def _get_original_tweet_id(self,tweet):
        raw_tweet = tweet.raw
        if 'referenced_tweets' in raw_tweet.keys():
            original_id = raw_tweet['referenced_tweets'][0]['id']
            return original_id
        return tweet.id