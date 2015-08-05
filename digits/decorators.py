from django_import import *
from functools import wraps
from flask import g, request, redirect, url_for

import redis
redis_conn = redis.StrictRedis(host='localhost', port=6379, db=0)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        djsession_id = request.cookies.get("sessionid")
        print djsession_id
        if djsession_id is None:
            print "Out from 1st condition"
            print djsession_id
            return redirect("http://127.0.0.1:8000/accounts/login", code = 302)

        # key = get_session_prefixed(djsession_id)
        key = djsession_id
        session_store = SessionStore(key)
        auth = session_store.load()

        if not auth:
            return redirect("http://127.0.0.1:8000/accounts/login", code = 302)

        g.user_id = str(auth.get("sessionid"))
        return f(*args, **kwargs)
    return decorated_function

