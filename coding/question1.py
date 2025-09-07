#  改错题，将输入字符子划分为x+dp 形式， x不含字母dp, 求划分的种类数
import sys

for line in sys.stdin:
    string = line.split()

ans = set()
string = str(string)[2:-2]
last = 0
for i in range(len(string)-2):
    if string[i] == 'd' and string[i+1] == 'p':
        ans.add(string[last:i])
        last = i+2

for item in ans:
    if 'dp' in item or 'p' in item:
        ans.remove(item)

print(len(ans))