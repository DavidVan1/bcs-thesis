import requests
from time import time, sleep
from typing import Optional
from lxml import html
from urllib.parse import urlparse, parse_qs
from time import sleep
import logging
from .logger import logger
from .InsulaAuthorizationApiConfig import InsulaAuthorizationApiConfig


class InsulaOpenIDConnect(InsulaAuthorizationApiConfig):
    def __init__(self, authorization_endpoint: str, token_endpoint: str, client_id: str, redirect_uri: str,
                 disable_ssl: bool = False, loglevel: str = 'INFO'):
        self._client_id: str = client_id
        self._redirect_uri: str = redirect_uri
        self._authorization_endpoint: str = authorization_endpoint
        self._token_endpoint: str = token_endpoint
        self._token_type: str = 'Bearer'
        self._scope: Optional[str] = 'openid'
        self._client_secret: Optional[str] = None

        self._username: Optional[str] = None
        self._password: Optional[str] = None

        self._init_token_time: int = 0
        self._init_refresh_token_time: int = 0

        self._refresh_token: Optional[str] = None
        self._access_token: Optional[str] = None

        self._expires_in = 0
        self._refresh_expires_in = 0

        self._session = requests.Session()
        if disable_ssl:
            self._session.verify = False

        if loglevel == 'CRITICAL':
            logging.root.level = logging.CRITICAL
        if loglevel == 'ERROR':
            logging.root.level = logging.ERROR
        if loglevel == 'WARNING':
            logging.root.level = logging.WARNING
        if loglevel == 'INFO':
            logging.root.level = logging.INFO
        if loglevel == 'DEBUG':
            logging.root.level = logging.DEBUG

    def set_user_credentials(self, username: str, password: str):
        self._username = username
        self._password = password

    def get_authorization_header(self):
        if not self._is_valid_token():
            attempt = 0
            while True:
                if attempt > 50:
                    raise RuntimeError('It is impossible to recover the token')
                try:
                    self._create_token()
                    break
                except Exception as e:
                    print(f'Errore durante la ricezione del token attempt: {attempt} of 50')
                    logger.error(e)

                    attempt += 1
                    sleep(10)

        return f'{self._token_type} {self._access_token}'

    def _is_refresh_token_valid(self, now) -> bool:
        return self._init_refresh_token_time + self._refresh_expires_in > now

    def _is_access_token_valid(self, now) -> bool:
        return self._init_token_time + self._expires_in > now

    def _is_valid_token(self) -> bool:

        now = time()

        if self._init_token_time == 0 and self._init_refresh_token_time == 0:
            return False

        if self._is_access_token_valid(now):
            logger.debug('use token cache')
            return True

        if self._is_refresh_token_valid(now):
            logger.debug('use refresh token cache')
            self._retrieve_refresh_token_from_endpoint()
            return True

        logger.debug('refresh token not valid')
        return False

    def _create_token(self):
        logger.debug('create token')
        return self._retrieve_token_from_endpoint(
            self._retrieve_authorization_from_endpoint()
        )

    def _retrieve_authorization_from_endpoint(self) -> str:
        """
        Calls the authorization endpoint and returns 'code'
        """
        get_params = {
            "client_id": self._client_id,
            "redirect_uri": self._redirect_uri,
            "scope": "openid",
            "response_type": "code"
        }

        response_auth = self._session.get(url=self._authorization_endpoint, params=get_params, timeout=(4100,400))
        if response_auth.status_code != 200:
            self._logs_and_raise('call authorization endpoint failed')

        auth_url = html.fromstring(response_auth.content.decode()).forms[0].action

        post_data = {
            "username": self._username,
            "password": self._password
        }

        response = self._session.post(auth_url, data=post_data, allow_redirects=False, timeout=(4100,400))

        if response.status_code != 302:
            self._logs_and_raise('Authorization failed, username or password incorrect [err: 199]')

        if 'Location' not in response.headers:
            self._logs_and_raise('Authorization failed, username or password incorrect [err: 100]')

        return parse_qs(urlparse(response.headers['Location']).query)['code'][0]

    def _retrieve_token_from_endpoint(self, code: str) -> dict:
        post_data = {
            "client_id": self._client_id,
            "redirect_uri": self._redirect_uri,
            "code": code,
            "grant_type": "authorization_code"
        }

        logger.debug("retrieve token")
        return self._get_token_from_server(post_data)

    def _retrieve_refresh_token_from_endpoint(self) -> dict:

        post_data = {
            "client_id": self._client_id,
            "redirect_uri": self._redirect_uri,
            "refresh_token": self._refresh_token,
            "grant_type": "refresh_token"
        }

        logger.debug("retrieve from refresh token")
        return self._get_token_from_server(post_data)

    def _get_token_from_server(self, data):
        response = self._session.post(self._token_endpoint, data=data, timeout=(4100,400))
        if response.status_code != 200:
            self._logs_and_raise('Cannot retrieve token [err: 110]')
        return self._parse_and_update_cache_token(response)

    def _parse_and_update_cache_token(self, response: requests.Response):
        _token = response.json()
        self._access_token = _token['access_token']
        self._refresh_token = _token['refresh_token']
        self._token_type = _token['token_type']
        self._expires_in = _token['expires_in']
        self._refresh_expires_in = _token['refresh_expires_in']
        self._init_token_time = time()
        self._init_refresh_token_time = time()
        return _token

    @staticmethod
    def _logs_and_raise(msg: str):
        logger.error(msg)
        raise Exception(msg)
