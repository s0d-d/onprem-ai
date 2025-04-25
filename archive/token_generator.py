from livekit import AccessToken

token = AccessToken(
    api_key="mykey",
    api_secret="mysecret",
    identity="sodoo",
    name="Sodoo",
)
token.add_grant(grant={"roomJoin": True, "room": "testroom"})

print(token.to_jwt())
