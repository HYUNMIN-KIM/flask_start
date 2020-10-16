
# ref : https://github.com/iinus/IMT4204-search-algorithms-in-IDS
# Input: two strings T (length n) and P (length m)
# Output: The locations of all occurences of P in T

class BNDM:

    def __init__(self):
        pass

    def create_bitmask_table(self, pattern, m):
        B = {-1: 0}
        m = len(pattern)
        s = 1
        for i in range(m - 1, -1, -1):
            B[pattern[i]] = B.get(pattern[i], 0) | s
            s <<= 1
        return B


    def searchString(self, text, pattern):
        n = len(text)
        m = len(pattern)
        # pre-processing
        B = self.create_bitmask_table(pattern, m)

        # search phase
        pos = 0
        positions = []  # store all positions found
        while (pos <= n - m):
            j = m - 1
            last = m
            d = ~0
            while (d != 0 and j >= 0):
                if text[pos + j] in B:
                    d = d & B[text[pos + j]]
                else:
                    d = d & 0
                j = j - 1
                if d != 0:
                    if (j >= 0):
                        last = j + 1
                    else:
                        positions.append(pos)
                d = d << 1
            pos = pos + last
        return positions

# bndm = BNDM()
# position = bndm.searchString("집에서 판교역가려면 몇번 버스타야돼요?", "판교역")
# #
# print(position)