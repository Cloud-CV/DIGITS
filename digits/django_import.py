import os, sys
sys.path.append('/home/pydev/Documents/Workspaces')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Workspaces.settings")
from django.conf import settings
from redis_sessions_fork.session import SessionStore

import os
import time
import subprocess
from threading import Thread
from flask import Flask, render_template, session, request
from flask.ext.socketio import SocketIO, emit, join_room, leave_room, \
    close_room, disconnect

# thread = None

import commands

# def get_download_status(socketio):
#     """
#         Method for calculating the percentage of completion
#         and then sending a respinse to page using socketIO.
#         This process is repeating untill the copy is not 
#         done completely. 
#     """
#     # dest_size = int(commands.getoutput('du -s '+dest_path).split()[0])
#     # source_size = int(source_size)
#     print "socketio is %s" %(socketio) 
#     dest_size = 1
#     source_size = 250
#     while(dest_size <= source_size):
#         # dest_size = int(commands.getoutput('du -s '+dest_path).split()[0])
#         time.sleep(0.01)
#         percent = (dest_size*100/source_size)
#         # dest_size = get_size(dest_path)
#         dest_size+=1
#         socketio.emit("progressbar",
#             {'percent':percent},namespace='/home')

