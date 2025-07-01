import redis
import threading
import time
def listen_and_delete(redis_host, redis_port, channel_name, other_redis_host, other_redis_port):
    # Connect to the Redis instance where keys will be deleted from
    r = redis.Redis(host=redis_host, port=redis_port)

    # Connect to the other Redis instance where keys will be deleted
    other_r = redis.Redis(host=other_redis_host, port=other_redis_port)

    # Subscribe to the specified channel
    pubsub = r.pubsub()
    pubsub.subscribe(channel_name)

    # Listening for messages in the channel
    for message in pubsub.listen():
        if message['type'] == 'message':
            key = message['data'].decode('utf-8')
            print("evicting")
            time.sleep(0.0001)
            #print("Received key " + key + " from Redis channel "+ channel_name +" on "+ redis_host+":"+redis_port +"Deleting from Redis...")

            # Delete the key from the current Redis database
            r.delete(key)

            # Delete the corresponding key from the other Redis database
            other_r.delete(key)

# Information for Redis instances
redis_info = [
    {
        'host': '127.0.0.1', 'port': 6388, 'channel': 'delete',
        'other_host': '127.0.0.1', 'other_port': 6378
    },
    {
        'host': '127.0.0.1', 'port': 6389, 'channel': 'delete',
        'other_host': '127.0.0.1', 'other_port': 6376
    },
    {
        'host': '127.0.0.1', 'port': 6390, 'channel': 'delete',
        'other_host': '127.0.0.1', 'other_port': 6380
    }
]

# Create threads to listen to each Redis channel and delete keys
threads = []
for info in redis_info:
    t = threading.Thread(
        target=listen_and_delete,
        args=(
            info['host'], info['port'], info['channel'],
            info['other_host'], info['other_port']
        )
    )
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()
