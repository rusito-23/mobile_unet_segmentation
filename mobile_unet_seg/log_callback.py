"""
Telegram + Logger Callback.
"""
from tensorflow.keras.callbacks import Callback
import os
import time
import requests
import logging

log = logging.getLogger('mobile_unet_seg')


class LoggerCallback(Callback):
    def __init__(self, tg_users=[]):
        self.token = os.environ.get('TG_TOKEN')
        self.chats = [os.environ.get('TG_CHAT_ID')]
        self.BOLD = '<b>{}</b>'
        self.ITALIC = '<i>{}</i>'
        self.BASE_URL = f'https://api.telegram.org/bot{self.token}'
        self.SEND_MESSAGE_URL = self.BASE_URL+'/sendMessage'
        self.GET_UPDATES_URL = self.BASE_URL+'/getUpdates'
        self.step = 1

    def on_epoch_begin(self, epoch, logs={}):
        self.start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        time_spent = time.time() - self.start
        title = self.BOLD.format(f"- EPOCH {self.step} -")
        body = f"""
    - Metrics: {logs}
    - Time spent: {self.time_spent(time_spent)}
        """
        message = title + '\n' + body

        log.debug(f"\n-- EPOCH {self.step} --"
                  f"- Metrics: {logs}"
                  f"- Time spent: {self.time_spent(time_spent)} \n")

        if self.token is not None:
            for chat in self.chats:
                # send
                params = {
                    'chat_id': chat,
                    'text': message.encode('ascii', 'replace'),
                    'parse_mode': 'html'}
                url = self.SEND_MESSAGE_URL
                self.get(url, params)
        self.step += 1

    def get(self, url, params):
        conn_error = False
        try:
            req = requests.get(url=url, params=params)
        except:
            conn_error = True
        if conn_error or req.status_code not in range(200, 300):
            log.error(f"[ERROR] sending notification: {params.get('chat_id')}")

    def time_spent(self, c):
        hours = c // 3600 % 24
        minutes = c // 60 % 60

        return f'{hours:.0f} hours and {minutes:.0f} minutes.'
