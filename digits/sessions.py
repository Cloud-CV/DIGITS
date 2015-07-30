# from django.contrib.sessions.backends.base import SessionBase
from django_import import * 
from redis_session_fork.session import *
class SessionStore(object):

    # The default serializer, for now
    def __init__(self, conn, session_key, secret, serializer=None):

        self._conn = conn
        self.session_key = session_key
        self._secret = secret
        self.serializer = serializer or JSONSerializer

    def load(self):
        session_data = self._conn.get(self.session_key)

        if not session_data is None:
            return self._decode(session_data)
        else:
            return {}

'''
    We only need the load method because it’s a read-only implementation of the storage.
    That means you can’t logout directly from Flask; instead, you might want to 
    redirect this task to Django. This is the reason for which the exists() and _decode() 
    methods are commented.

''' 
    # def exists(self, session_key):
    #     return self._conn.exists(session_key)


    # def _decode(self, session_data):
    #     """
    #     Decodes the Django session
    #     :param session_data:
    #     :return: decoded data
    #     """
    #     encoded_data = base64.b64decode(force_bytes(session_data))
    #     try:
    #         # Could produce ValueError if there is no ':'
    #         hash, serialized = encoded_data.split(b':', 1)
    #         # In the Django version of that they check for corrupted data
    #         # I don't find it useful, so I'm removing it
    #         return self.serializer().loads(serialized)
    #     except Exception as e:
    #         # ValueError, SuspiciousOperation, unpickling exceptions. If any of
    #         # these happen, return an empty dictionary (i.e., empty session).
    #         return {}
