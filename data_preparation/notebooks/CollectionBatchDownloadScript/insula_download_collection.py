import os
import json
import requests
from pathlib import Path

import config as CONFIG
import auth as AUTH
from InsulaWorkflowClient import InsulaOpenIDConnect


class InsulaClientDownloader:
    def __init__(self):
        self.session = requests.session()

        self.insulaAuth: InsulaOpenIDConnect = InsulaOpenIDConnect(
            authorization_endpoint=CONFIG.Download.authorization_endpoint,
            token_endpoint=CONFIG.Download.token_endpoint,
            redirect_uri=CONFIG.Download.redirect_uri,
            client_id=CONFIG.Download.client_id
        )
        self.insulaAuth.set_user_credentials(username=AUTH.Insula.username, password=AUTH.Insula.password)
        self.bearer = self.insulaAuth.get_authorization_header()

    def __del__(self):
        self.session.close()

    def download_file(self, dl_link, filename):
        self.bearer = self.insulaAuth.get_authorization_header()

        print(dl_link + " " + filename)

        response = self.session.get(dl_link, headers={'Authorization': self.bearer})

        if response.status_code >= 400:
            raise Exception(f"Authentication failed with status: {response.status_code}")

        filepath = f'{CONFIG.download_path}/{filename}'
        try:
            with open(filepath, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(e)
        return filepath

    def get_file_list(self, collection_id):
        self.bearer = self.insulaAuth.get_authorization_header()
        resp = self.session.get(
            CONFIG.Download.base_url + "/secure/api/v2.0/platformFiles/search/parametricFind?"
                                       "collection=https://insula.earth/secure/api/v2.0/collections/"
            + str(collection_id) + "&page=0&size=1000",
            headers={'Authorization': self.bearer}
        )
        print(resp.text)
        return json.loads(resp.text)


def insula_download_collection():
    os.makedirs(os.path.dirname(CONFIG.donefile), exist_ok=True)
    os.makedirs(CONFIG.download_path, exist_ok=True)

    Path(CONFIG.donefile).touch()
    Path(CONFIG.failedfile).touch()

    downloader = InsulaClientDownloader()

    collection_id = 7
    list_json = downloader.get_file_list(collection_id)

    for pf_entry in list_json['_embedded']['platformFiles']:
        print(pf_entry['filename'])

        dl_link = pf_entry['_links']['download']['href']
        filename = pf_entry['filename'].rsplit('/', 1)[1]

        with open(CONFIG.donefile) as list_done:
            content = list_done.read()

        if filename not in content:
            try:
                # DOWNLOAD FILE AND METADATA FROM OPE
                downloader = InsulaClientDownloader()
                downloader.download_file(dl_link, filename)

                # ADD TO THE COMPLETED LIST
                with open(CONFIG.donefile, "a") as list_done:
                    list_done.write(filename + "\n")

            except Exception as e:
                print(f"Error during download of file {filename}: {e}")
                # ADD TO THE FAILED LIST IF SOMETHING WENT WRONG
                with open(CONFIG.failedfile, "a") as list_failed:
                    list_failed.write(filename + "\n")


if __name__ == '__main__':
    insula_download_collection()
