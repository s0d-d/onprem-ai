# server.py
import os
from livekit import api
from flask import Flask

app = Flask(__name__)

@app.route('/getToken')
def getToken():
  token = api.AccessToken(os.getenv('LIVEKIT_API_KEY'), os.getenv('LIVEKIT_API_SECRET')) \
    .with_identity("sodoo") \
    .with_name("Sodoo") \
    .with_grants(api.VideoGrants(
        room_join=True,
        room="my-room",
    ))
  return token.to_jwt()


print("Starting Flask token server...")

if __name__ == "__main__":
    app.run(debug=True)