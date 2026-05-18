collection_id = 7

download_path = "downloaded"
donefile = "lists/done.txt"
failedfile = "lists/failed.txt"


class Download:
    authorization_endpoint="https://identity.insula.earth/realms/phisat2/protocol/openid-connect/auth"
    token_endpoint="https://identity.insula.earth/realms/phisat2/protocol/openid-connect/token"
    redirect_uri="http://localhost:9207/auth"
    client_id="api-client"
    base_url="https://phisat2.insula.earth"
