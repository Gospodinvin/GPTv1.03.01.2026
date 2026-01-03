import time
import asyncio

class TTLState:
    def __init__(self, ttl):
        self.ttl = ttl
        self.data = {}
        self.lock = asyncio.Lock()

    async def set(self, uid, key, val):
        async with self.lock:
            if uid not in self.data:
                self.data[uid] = {"_exp": time.time() + self.ttl}
            self.data[uid]["_exp"] = time.time() + self.ttl
            self.data[uid][key] = val

    async def get(self, uid, key):
        async with self.lock:
            if uid not in self.data:
                return None
            d = self.data[uid]
            if d["_exp"] < time.time():
                del self.data[uid]
                return None
            return d.get(key)

    async def clear(self, uid):
        async with self.lock:
            self.data.pop(uid, None)

    async def get_all(self, uid):
        async with self.lock:
            if uid not in self.data or self.data[uid]["_exp"] < time.time():
                self.data.pop(uid, None)
                return {}
            return {k: v for k, v in self.data[uid].items() if k != "_exp"}
