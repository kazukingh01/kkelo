import string, time
import numpy as np
from kkelo import Elo


N        = 10000
ndfstr   = np.array(list(string.ascii_lowercase))
ndfwk    = np.random.randint(0, len(ndfstr), (5000, 10))
ndfpl    = ndfstr[ndfwk]
ndfpl    = np.unique(ndfpl[:, 0] + ndfpl[:, 1] + ndfpl[:, 2])
ndfpl    = [str(x) for x in ndfpl]
list_rdm = [np.random.permutation(len(ndfpl))[:6]   for _ in range(N)]
list_rnk = [(np.random.permutation(6) + 1).tolist() for _ in range(N)]
timest   = time.time()
elo1     = Elo(is_check=True)
elo1.add_players(ndfpl)
for i in range(N):
    players = [ndfpl[x] for x in list_rdm[i]]
    elo1.update(*players, ranks=list_rnk[i])
print(time.time() - timest)
timest  = time.time()
elo2    = Elo(is_check=False)
elo2.add_players(ndfpl)
for i in range(N):
    players = [ndfpl[x] for x in list_rdm[i]]
    elo2.update(*players, ranks=list_rnk[i])
print(time.time() - timest)
listwk = [[str(y) for y in x] for x in np.array_split(np.array(ndfpl), 1000)]
timest = time.time()
ndf1   = np.concatenate(elo1.ratings(*ndfpl)[1])
ndf1   = np.array(elo1.ratings_team(*listwk))
print(time.time() - timest)
timest = time.time()
ndf2   = np.concatenate(elo2.ratings(*ndfpl)[1])
ndf2   = np.array(elo2.ratings_team(*listwk))
print(time.time() - timest)
elo1.evaluate(*listwk[:6], ranks=[1,2,3,4,5,6])
elo2.evaluate(*listwk[:6], ranks=[1,2,3,4,5,6])
ndfwk = np.array(ndfpl)
elo1.evaluate(ndfwk[:1000].reshape(-1, 10), ranks=(np.argsort(np.random.rand(100, 10)) + 1))
elo2.evaluate(ndfwk[:1000].reshape(-1, 10), ranks=(np.argsort(np.random.rand(100, 10)) + 1))
elo1.evaluate(ndfwk[:1000].reshape(-1, 10), ranks=(np.argsort(np.random.rand(100,  5)) + 1), structure=[2, 5, 6, 8])
elo2.evaluate(ndfwk[:1000].reshape(-1, 10), ranks=(np.argsort(np.random.rand(100,  5)) + 1), structure=[2, 5, 6, 8])
