import contextlib
import io
import json
import unittest
import websocket
from jlab.config import JlabConfig
from jlab.exceptions import KernelError
from jlab.kernel import KernelConnection


class Socket:
    def __init__(self, frames):
        self.frames=iter(frames)
    def send(self, raw):
        self.message_id=json.loads(raw)['header']['msg_id']
    def settimeout(self, timeout):
        pass
    def recv(self):
        try:kind,content,own=next(self.frames)
        except StopIteration:raise websocket.WebSocketTimeoutException()
        return json.dumps(dict(header={'msg_type':kind},content=content,
            parent_header={'msg_id':self.message_id if own else 'another-request'}))


REPLY=('execute_reply',{'status':'ok'},True)
IDLE=('status',{'execution_state':'idle'},True)
STREAM=('stream',{'name':'stdout','text':'process is alive\n'},True)


class KernelOutputOrderTests(unittest.TestCase):
    def execute(self, method, frames):
        connection=KernelConnection(JlabConfig(url='https://example.invalid',token=''), 'test')
        connection._ws=Socket(frames)
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            return getattr(connection,method)('print("process is alive")',timeout=.1)

    def test_output_after_reply_is_not_lost(self):
        for method in ('execute','execute_streaming'):
            with self.subTest(method=method):
                result=self.execute(method,[REPLY,STREAM,IDLE])
                self.assertEqual(result.outputs,[{'type':'stream','name':'stdout','text':'process is alive\n'}])

    def test_idle_before_reply_and_silent_execution(self):
        for method in ('execute','execute_streaming'):
            with self.subTest(method=method):
                self.assertEqual(len(self.execute(method,[STREAM,IDLE,REPLY]).outputs),1)
                self.assertEqual(self.execute(method,[REPLY,IDLE]).outputs,[])

    def test_both_completion_messages_must_belong_to_the_request(self):
        for method in ('execute','execute_streaming'):
            for frames in ([IDLE],[REPLY],[(IDLE[0],IDLE[1],False),REPLY]):
                with self.subTest(method=method,frames=frames):
                    with self.assertRaises(KernelError):self.execute(method,frames)

    def test_error_reply_retains_later_output(self):
        error=('execute_reply',{'status':'error','ename':'ValueError','evalue':'bad input','traceback':['trace']},True)
        for method in ('execute','execute_streaming'):
            with self.subTest(method=method):
                result=self.execute(method,[error,STREAM,IDLE])
                self.assertEqual((result.status,result.error_name,result.error_value),('error','ValueError','bad input'))
                self.assertEqual(result.traceback,['trace'])
                self.assertEqual(len(result.outputs),1)


if __name__=='__main__':unittest.main()
