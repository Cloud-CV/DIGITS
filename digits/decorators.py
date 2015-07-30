from django_import import *
from functools import wraps
from flask import g, request, redirect, url_for

import redis
redis_conn = redis.StrictRedis(host='localhost', port=6379, db=0)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        djsession_id = request.cookies.get("sessionid")
        if djsession_id is None:
            return redirect("/")

        key = get_session_prefixed(djsession_id)
        session_store = SessionStore(redis_conn, key)
        auth = session_store.load()

        if not auth:
            return redirect("/login")

        g.user_id = str(auth.get("_auth_user_id"))

        return f(*args, **kwargs)
    return decorated_function

