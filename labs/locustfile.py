"""The load of Lab 4.3: members' requests, sent to one of the two adapters.

    locust -f locustfile.py --headless -u 8 -r 8 -t 20s --host http://127.0.0.1:8310 --csv artifacts/4.3/u8

Each simulated user sends a request, waits for the answer and sends the next one at once, so
the number of users is the number of requests in flight.
"""

import json
import os
import random
from pathlib import Path

from locust import HttpUser, constant, task

ROWS = [json.loads(line) for line in (Path(__file__).parent / "data" / "requests_val.jsonl").read_text(encoding="utf-8").splitlines()]
MODELS = os.environ.get("LOAD_MODELS", "qayd-registry,qayd-replay").split(",")


class Member(HttpUser):
    wait_time = constant(0)

    @task
    def register(self):
        row = random.choice(ROWS)
        body = {"model": random.choice(MODELS), "messages": [{"role": "user", "content": row["text"]}]}
        with self.client.post("/v1/chat/completions", json=body, name="chat", catch_response=True) as response:
            if response.status_code != 200 or "<tool_call>" not in response.text:
                response.failure("no tool call in the answer")
