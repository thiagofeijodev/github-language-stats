import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from github_stats import Queries, Stats


class FakeResponse:
    def __init__(self, status, payload, headers=None):
        self.status = status
        self._payload = payload
        self.headers = {} if headers is None else headers

    async def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response):
        self.response = response

    async def post(self, *args, **kwargs):
        return self.response

    async def get(self, *args, **kwargs):
        if isinstance(self.response, list):
            return self.response.pop(0)
        return self.response


class FakeQueries:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def query(self, generated_query):
        self.calls += 1
        await asyncio.sleep(0)
        return self.response


class QueriesTests(unittest.IsolatedAsyncioTestCase):
    async def test_graphql_errors_are_reported_without_response_details(self):
        private_repo = "owner/private-repository"
        response = FakeResponse(
            200,
            {"data": {}, "errors": [{"message": f"Denied for {private_repo}"}]},
        )
        queries = Queries("user", "token", FakeSession(response))

        with self.assertRaisesRegex(RuntimeError, "GraphQL API returned errors") as error:
            await queries.query("query { viewer { login } }")

        self.assertNotIn(private_repo, str(error.exception))

    async def test_graphql_http_error_is_reported(self):
        response = FakeResponse(401, {})
        response.json = AsyncMock(side_effect=AssertionError("body must not be read"))
        queries = Queries("user", "token", FakeSession(response))

        with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
            await queries.query("query { viewer { login } }")

    async def test_rest_http_error_is_reported(self):
        queries = Queries("user", "token", FakeSession(FakeResponse(403, {})))

        with self.assertRaisesRegex(RuntimeError, "HTTP 403"):
            await queries.query_rest("/repos/owner/private-repository/stats/contributors")

    async def test_empty_contributor_statistics_return_empty_list(self):
        response = FakeResponse(204, None)
        response.json = AsyncMock(side_effect=AssertionError("body must not be read"))
        queries = Queries("user", "token", FakeSession(response))

        result = await queries.query_rest("/repos/owner/project/stats/contributors")

        self.assertEqual(result, [])

    async def test_unavailable_traffic_views_are_omitted(self):
        queries = Queries("user", "token", FakeSession(FakeResponse(404, {})))

        result = await queries.query_rest("/repos/owner/project/traffic/views")

        self.assertEqual(result, {})

    async def test_traffic_permission_denial_is_omitted(self):
        queries = Queries("user", "token", FakeSession(FakeResponse(403, {})))

        result = await queries.query_rest("/repos/owner/project/traffic/views")

        self.assertEqual(result, {})

    async def test_rest_rate_limit_is_reported(self):
        response = FakeResponse(403, {}, {"X-RateLimit-Remaining": "0"})
        queries = Queries("user", "token", FakeSession(response))

        with self.assertRaisesRegex(RuntimeError, "rate limit exceeded"):
            await queries.query_rest("/repos/owner/project/stats/contributors")

    async def test_rest_retries_empty_202_response(self):
        accepted = FakeResponse(202, None)
        accepted.json = AsyncMock(side_effect=AssertionError("body must not be read"))
        session = FakeSession([accepted, FakeResponse(200, {"views": []})])
        queries = Queries("user", "token", session)

        with patch("github_stats.asyncio.sleep", new_callable=AsyncMock) as sleep:
            result = await queries.query_rest("/repos/user/project/traffic/views")

        self.assertEqual(result, {"views": []})
        sleep.assert_awaited_once_with(2)


class StatsTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_repository_data_is_not_treated_as_empty_stats(self):
        stats = Stats("user", "token", session=None)
        stats.queries = FakeQueries({"data": {"viewer": {"login": "user"}}})

        with self.assertRaisesRegex(RuntimeError, "omitted repository data"):
            await stats.get_stats()

    async def test_concurrent_stats_load_once_and_aggregate(self):
        response = {
            "data": {
                "viewer": {
                    "login": "user",
                    "name": "User",
                    "repositories": {
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": [
                            {
                                "nameWithOwner": "user/project",
                                "stargazers": {"totalCount": 3},
                                "forkCount": 2,
                                "languages": {
                                    "edges": [
                                        {
                                            "size": 120,
                                            "node": {
                                                "name": "Python",
                                                "color": "#3572A5",
                                            },
                                        }
                                    ]
                                },
                            }
                        ],
                    },
                    "repositoriesContributedTo": {
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": [],
                    },
                }
            }
        }
        stats = Stats("user", "token", session=None)
        fake_queries = FakeQueries(response)
        stats.queries = fake_queries

        await asyncio.gather(stats.get_stats(), stats.get_stats())

        self.assertEqual(fake_queries.calls, 1)
        self.assertEqual(await stats.stargazers, 3)
        self.assertEqual(await stats.forks, 2)
        self.assertEqual((await stats.languages)["Python"]["prop"], 100)