import numpy as np
from sklearn.linear_model import SGDRegressor

def get_binary_encodings(n):
    ans = []
    low = (1<<n)
    high = (low << 1)

    for tx in range(low,high):
        tans = [-1]*n
        for j in range(n):
            if tx & (1<<j) :
                tans[j]=1
        ans.append(tans)
    return ans

# def

if __name__ == "__main__":
    l = np.array(get_binary_encodings(8))
    print(l.shape)