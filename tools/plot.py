#299 57
#{0: 116, 1: 104, 2: 16, 3: 63}
#{0: 8, 1: 21, 2: 9, 3: 19}
import matplotlib.pyplot as plt
y = [116, 104, 16, 8, 21, 9]
x = ['macro\npos', 'macro\nneg', 'macro\nsur', 'micro\npos', 'micro\nneg', 'micro\nsur']
plt.bar(x, y)
for a,b in zip(x, y):
    plt.text(a, b+1, str(b))
plt.title('Major and minor types')
plt.show()