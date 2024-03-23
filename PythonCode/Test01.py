# Writer：TuTTTTT
# 编写Time:2024/3/22 11:41
numer = [1, 2, 3, 4, 5]
x = None
n = [(x := num) ** 2 for num in numer if (x := num) % 2 == 0]
# 容易出现NameError和TypeError
print(n)
