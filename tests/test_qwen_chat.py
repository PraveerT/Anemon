import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from jlab.cli import main
from jlab.config import JlabConfig
from jlab import qwen_chat


class FakeRemote:
    def __init__(self, answer="reply", error=None):
        self.answer = answer
        self.error = error
        self.connect_calls = 0
        self.ensure_calls = 0
        self.complete_calls = []
        self.invalidate_calls = 0
        self.mirrors = []

    def connect(self):
        self.connect_calls += 1

    def ensure_service(self):
        self.ensure_calls += 1

    def complete(self, messages, **settings):
        self.complete_calls.append((messages, settings))
        if self.error:
            raise self.error
        return self.answer

    def invalidate_bridge(self):
        self.invalidate_calls += 1

    def mirror_history(self, name, messages):
        self.mirrors.append((name, messages))


class QwenStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.state_patch = patch.object(
            qwen_chat, "STATE_DIR", Path(self.temporary.name) / "qwen"
        )
        self.state_patch.start()

    def tearDown(self):
        self.state_patch.stop()
        self.temporary.cleanup()

    def test_config_and_history_are_persisted_atomically(self):
        config = qwen_chat.QwenConfig(model_id="example/model", port=9876)
        qwen_chat.save_qwen_config(config)
        qwen_chat.save_history(
            "research",
            [{"role": "user", "content": "hello"}],
        )

        self.assertEqual(qwen_chat.load_qwen_config(), config)
        self.assertEqual(
            qwen_chat.load_history("research"),
            [{"role": "user", "content": "hello"}],
        )
        self.assertEqual(list(qwen_chat.STATE_DIR.rglob("*.tmp")), [])
        self.assertEqual(
            json.loads((qwen_chat.STATE_DIR / "config.json").read_text()),
            qwen_chat.asdict(config),
        )

    def test_failed_generation_does_not_append_dangling_user_turn(self):
        qwen_chat.save_history(
            "default",
            [{"role": "assistant", "content": "earlier"}],
        )
        remote = FakeRemote(error=RuntimeError("transport lost"))
        chat = qwen_chat.QwenChat(qwen_chat.QwenConfig(), remote)

        with self.assertRaises(qwen_chat.QwenError):
            chat.send("default", "new question")

        self.assertEqual(
            qwen_chat.load_history("default"),
            [{"role": "assistant", "content": "earlier"}],
        )
        self.assertEqual(remote.complete_calls.__len__(), 2)
        self.assertEqual(remote.invalidate_calls, 1)

    def test_success_commits_complete_turn_and_mirrors_it(self):
        remote = FakeRemote(answer="answer")
        chat = qwen_chat.QwenChat(qwen_chat.QwenConfig(), remote)

        answer, warning = chat.send("work", "question", system="be concise")

        expected = [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        self.assertEqual(answer, "answer")
        self.assertIsNone(warning)
        self.assertEqual(qwen_chat.load_history("work"), expected)
        self.assertEqual(remote.mirrors, [("work", expected)])

    def test_prompt_is_only_embedded_as_base64_json(self):
        prompt = "dangerous ' prompt\n__import__('os').system('no')"
        code = qwen_chat._payload_prelude({"prompt": prompt})

        self.assertNotIn(prompt, code)
        encoded = code.split("b64decode('", 1)[1].split("')", 1)[0]
        self.assertEqual(
            json.loads(qwen_chat.base64.b64decode(encoded)), {"prompt": prompt}
        )


class QwenRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.state_patch = patch.object(
            qwen_chat, "STATE_DIR", Path(self.temporary.name) / "qwen"
        )
        self.state_patch.start()

    def tearDown(self):
        self.state_patch.stop()
        self.temporary.cleanup()

    @patch.object(qwen_chat, "fetch_running_notebook")
    @patch.object(qwen_chat, "load_ps_api_key", return_value="key")
    @patch.object(qwen_chat, "save_config")
    @patch.object(qwen_chat, "load_config")
    @patch.object(qwen_chat, "JupyterClient")
    def test_stale_endpoint_refreshes_from_running_notebook_without_starting(
        self, client_class, load_config_mock, save_config_mock, _key, fetch
    ):
        saved = JlabConfig("https://stale.test", "old")
        load_config_mock.return_value = saved
        stale_client = Mock()
        stale_client.status.side_effect = RuntimeError("gone")
        live_client = Mock()
        client_class.side_effect = [stale_client, live_client]
        fetch.return_value = {
            "url": "https://fresh.test",
            "token": "new",
            "id": "nb",
            "name": "gpu",
        }

        with patch.object(qwen_chat, "ps_start_notebook") as start:
            remote = qwen_chat.QwenRemote(qwen_chat.QwenConfig())
            connected = remote.connect()

        self.assertIs(connected, live_client)
        start.assert_not_called()
        save_config_mock.assert_called_once()
        self.assertEqual(save_config_mock.call_args.args[0].url, "https://fresh.test")

    @patch.object(qwen_chat.time, "sleep")
    @patch.object(qwen_chat, "ps_start_notebook")
    @patch.object(qwen_chat, "fetch_running_notebook")
    @patch.object(qwen_chat, "load_ps_api_key", return_value="key")
    @patch.object(qwen_chat, "save_config")
    @patch.object(qwen_chat, "load_config")
    @patch.object(qwen_chat, "JupyterClient")
    def test_stopped_notebook_is_started_then_polled(
        self,
        client_class,
        load_config_mock,
        _save,
        _key,
        fetch,
        start,
        _sleep,
    ):
        load_config_mock.return_value = JlabConfig("https://stale.test", "old")
        stale_client = Mock()
        stale_client.status.side_effect = RuntimeError("gone")
        live_client = Mock()
        client_class.side_effect = [stale_client, live_client]
        fetch.side_effect = [None, None, {
            "url": "https://fresh.test",
            "token": "new",
            "id": "nb",
            "name": "gpu",
        }]

        remote = qwen_chat.QwenRemote(qwen_chat.QwenConfig())
        remote.connect()

        start.assert_called_once_with("key")
        self.assertEqual(fetch.call_count, 3)

    @patch.object(qwen_chat, "fetch_running_notebook")
    @patch.object(qwen_chat, "load_ps_api_key")
    @patch.object(qwen_chat, "load_config")
    @patch.object(qwen_chat, "JupyterClient")
    def test_live_endpoint_is_fast_path(
        self, client_class, load_config_mock, load_key, fetch
    ):
        config = JlabConfig("https://live.test", "token")
        load_config_mock.return_value = config
        live_client = Mock()
        client_class.return_value = live_client

        remote = qwen_chat.QwenRemote(qwen_chat.QwenConfig())
        self.assertIs(remote.connect(), live_client)

        client_class.assert_called_once_with(
            config, request_timeout=qwen_chat.QWEN_HTTP_TIMEOUT
        )
        load_key.assert_not_called()
        fetch.assert_not_called()

    def test_healthy_service_is_not_started_again(self):
        remote = qwen_chat.QwenRemote(qwen_chat.QwenConfig())
        remote.health = Mock(return_value={
            "healthy": True,
            "provisioned": True,
            "service_model": qwen_chat.DEFAULT_MODEL,
            "service_version": qwen_chat.REMOTE_SERVER_VERSION,
            "service_max_model_len": qwen_chat.QwenConfig().max_model_len,
        })
        remote._execute = Mock()

        remote.ensure_service()

        remote._execute.assert_not_called()

    def test_changed_service_settings_trigger_restart(self):
        remote = qwen_chat.QwenRemote(qwen_chat.QwenConfig(max_model_len=8192))
        remote.health = Mock(return_value={
            "healthy": True,
            "provisioned": True,
            "service_model": qwen_chat.DEFAULT_MODEL,
            "service_version": qwen_chat.REMOTE_SERVER_VERSION,
            "service_max_model_len": 16384,
        })
        remote._execute = Mock(return_value=Mock(
            outputs=[{
                "type": "stream",
                "text": qwen_chat.SENTINEL + qwen_chat._encoded_payload({
                    "healthy": True,
                }) + "\n",
            }],
            status="ok",
        ))

        remote.ensure_service()

        code = remote._execute.call_args.args[0]
        encoded = code.split("b64decode('", 1)[1].split("')", 1)[0]
        payload = json.loads(qwen_chat.base64.b64decode(encoded))
        self.assertTrue(payload["restart_service"])

    def test_remote_server_disables_visible_thinking(self):
        self.assertIn('"enable_thinking": False', qwen_chat.REMOTE_SERVER)
        self.assertIn('re.sub(r"(?s)^.*?</think>', qwen_chat.REMOTE_SERVER)


class QwenCliTests(unittest.TestCase):
    def test_qwen_group_and_commands_are_registered(self):
        runner = CliRunner()

        root = runner.invoke(main, ["--help"])
        group = runner.invoke(main, ["qwen", "--help"])

        self.assertEqual(root.exit_code, 0, root.output)
        self.assertIn("qwen", root.output)
        self.assertEqual(group.exit_code, 0, group.output)
        for command in ("setup", "chat", "status", "history"):
            self.assertIn(command, group.output)

    @patch.object(qwen_chat, "_clear_qwen_session")
    @patch.object(qwen_chat, "QwenRemote")
    def test_status_uses_a_fresh_bridge(self, remote_class, clear_session):
        remote = remote_class.return_value
        remote.health.return_value = {"provisioned": True, "healthy": False}

        result = CliRunner().invoke(main, ["qwen", "status"])

        self.assertEqual(result.exit_code, 0, result.output)
        remote.connect.assert_called_once_with(start_if_stopped=False)
        clear_session.assert_called_once_with()
        remote.health.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
