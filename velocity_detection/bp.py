import random, json
from math import atan as arctan


def forward(x, params, return_all=False):
    A, B, C, D = params
    z1 = B * x + C
    a1 = arctan(z1)
    z2 = A * a1 + D
    a2 = z2

    if return_all:
        return [z1, a1, z2, a2]
    else:
        return a2

def cost(data, params, is_random=False, n=None):
    if is_random:
        using_data = random.shuffle(data)[0:n]
    else:
        using_data = data
        n = len(data)

    return sum([(forward(using_data[i][0], params) - using_data[i][1]) ** 2 
                for i in range(n)]) / (2 * n)

def backward(x, y, params, eta):
    A, B, C, D = params
    z1, a1, z2, a2 = forward(x, params, return_all=True)

    delta2 = forward(x, params) - y
    delta1 = A * delta2 / (1 + z1 ** 2)

    return [
        A - eta * a1 * delta2, 
        B - eta * x * delta1, 
        C - eta * delta1, 
        D - eta * delta2
    ]

def train(data, epochs=100000, params=None, eta=0.01):
    if not params:
        params = [random.gauss(mu=100, sigma=1000) for i in range(4)]
    eta = 0.01
    for epoch in range(epochs):
        for x, y in data:
            params = backward(x, y, params, eta)
        if epoch % 10000 == 0:
            c = cost(data, params)
            print(f"{epoch}/{epochs}, cost is {c}")
    return params


if __name__ == "__main__":

    PATH = "velocity_detection/training_data.json"
    
    with open(PATH) as training_file:
        data = training_file.read()
    data = json.loads(data)

    params = train(data)
    print(params)
    print(cost(data, params))

