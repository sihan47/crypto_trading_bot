import unittest
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from strategies import btc_news_summary as news


class NewsSummaryTests(unittest.TestCase):
    def setUp(self):
        news._NEWS_CACHE.clear()

    @patch.object(news, '_news_config', return_value={'model': 'test', 'cache_seconds': 1800})
    @patch.object(news, 'fetch_news', return_value=['A headline'])
    @patch.object(news, 'summarize_with_openai', return_value='A digest')
    def test_two_symbols_share_cached_summary(self, summarize, fetch, config):
        self.assertEqual(news.get_news_summary(), '[NEWS] A digest')
        self.assertEqual(news.get_news_summary(), '[NEWS] A digest')
        fetch.assert_called_once()
        summarize.assert_called_once()

    @patch.object(news, '_news_config', return_value={})
    @patch.object(news, 'fetch_news', side_effect=RuntimeError('secret-in-error'))
    def test_failure_does_not_leak_error_and_is_temporarily_cached(self, fetch, config):
        result = news.get_news_summary()
        self.assertIn('Unavailable', result)
        self.assertNotIn('secret', result)
        news.get_news_summary()
        fetch.assert_called_once()

    @patch.object(news, 'OpenAI')
    def test_truncated_response_is_not_accepted(self, client):
        client.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason='length', message=SimpleNamespace(content='partial'))]
        )
        with self.assertRaises(RuntimeError):
            news.summarize_with_openai(['headline'])

    def test_news_provenance_and_staleness(self):
        response=SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'status':'ok','articles':[{'title':'Headline','publishedAt':'2026-11-15T10:00:00Z','source':{'name':'Publisher'},'url':'https://example.org/story'}]})
        with patch.object(news,'NEWS_API_KEY','test'),patch.object(news.requests,'get',return_value=response):
            items=news.fetch_news()
        self.assertIn('https://example.org/story',items[0])
        self.assertEqual(news.get_news_context()['articles'][0]['source'],'Publisher')
        news._NEWS_META['retrieved_at']=(datetime.now(timezone.utc)-timedelta(hours=2)).isoformat()
        self.assertEqual(news.get_news_context()['status'],'stale')


if __name__ == '__main__':
    unittest.main()
