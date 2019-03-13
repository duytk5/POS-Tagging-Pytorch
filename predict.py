import math

"""
input : y_hat - len * dim
"""


def greedy_predict(y_hat):
    length = y_hat.shape[0]
    dim = y_hat.shape[1]
    ans = []
    for i in range(length):
        ma = y_hat[i][0]
        for j in range(1, dim):
            ma = max(ma, y_hat[i][j])
        ans.append(ma)
    return ans


def beam_search_predict(y_hat, k):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in y_hat:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -math.log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences
