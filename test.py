from __future__ import print_function

f = open("./resources/embedding/baomoi-050.bin.txt", "r")

g = open("./resources/embedding/word2vec_vi.txt", "w")

id = 0
n = 0
dim = 0
sl = 0
for line in f:
    if id == 0:
        n = int(line.replace("\n", "").replace("\xef", "").replace("\xbb", "").replace("\xbf", "").replace("\r", ""))
    if id == 1:
        dim = int(line.replace("\n", "").replace("\xef", "").replace("\xbb", "").replace("\xbf", "").replace("\r", ""))
    if id == 2:
        print(str(433155) + " " + str(dim), file=g)
    if id > 1:
        line = line.replace("\n", "").replace("\r", "")
        ll = line.split(" ")
        ss = ""
        ok = True
        for i in range(len(ll)):
            if (len(ll)!= 51):
                ok = False
                break
            if i == 0:
                try:
                    cc = ll[0].decode('utf-8')
                except:
                    pass
                    ok = False

                ss = ss +ll[0] + " "
            else:
                ss = ss + str(float(ll[i])) + " "
        try:
            if ok :
                sl += 1
                print(ss , file=g)

        except:
            pass
    id += 1
print(sl)
