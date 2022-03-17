import os
from flask import Flask
from flask import send_file
from wsgiref.simple_server import make_server

WEB_DIST_DIR = 'web/dist/web'
app = Flask("__name__", static_folder=os.path.join(os.path.dirname(__file__), WEB_DIST_DIR),
            static_url_path='')


@app.route('/')
def index():
    return send_file(os.path.join(WEB_DIST_DIR, 'index.html'))


if __name__ == '__main__':
    server = make_server('127.0.0.1', 5000, app)
    server.serve_forever()
