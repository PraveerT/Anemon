import unittest
from unittest.mock import patch

from jlab.config import ps_start_notebook


class PaperspaceConfigTests(unittest.TestCase):
    @patch("requests.post")
    def test_start_uses_free_a6000_for_six_hours(self, post):
        response = post.return_value
        response.status_code = 200
        response.json.return_value = {
            "handle": "notebook-id",
            "fqdn": "example.test",
            "token": "token",
        }

        ps_start_notebook("api-key")

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["machineType"], "Free-A6000")
        self.assertEqual(payload["shutdownTimeout"], "6")


if __name__ == "__main__":
    unittest.main()
