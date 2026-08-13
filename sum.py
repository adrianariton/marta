import random


def valid(s: str) -> bool:
    last_d = s.rfind("d")
    last_r = s.rfind("r")
    last_w = s.rfind("w")

    # 'd' finishes first if its last occurrence is before the first 'w'
    return (last_d < last_w) and (last_d < last_r)


def sim():
    arr = ["d"] * 10 + ["r"] * 20 + ["w"] * 30
    s = ""
    for i in range(60):
        ball = random.choice(arr)
        arr.remove(ball)
        s += ball
    return 1 if valid(s) else 0


RR = 100000

p = 0
for k in range(RR):
    p += sim()

print(p / RR)
