import mongoengine as me
from datetime import datetime

import pytz


class ChatMessage(me.Document):
    user_id = me.StringField(required=True, max_length=100)
    role = me.StringField(required=True, choices=["user", "assistant"])
    message = me.StringField(required=True)
    timestamp = me.DateTimeField(default=lambda: datetime.now(pytz.utc))

    meta = {
        'collection': 'chat_messages',
        'ordering': ['-timestamp'],
    }