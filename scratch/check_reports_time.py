import os, time

now = time.time()
two_days = 2 * 86400
for root, dirs, files in os.walk('reports'):
    for f in files:
        p = os.path.join(root, f)
        mtime = os.path.getmtime(p)
        if now - mtime < two_days:
            print(f"{p}: {time.ctime(mtime)}")
