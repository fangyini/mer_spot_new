import matplotlib.pyplot as plt
x=['positive', 'negative', 'surprise', 'others']
y = [8, 21, 9, 19]
bars = plt.bar(x, y)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval + .05, yval)
plt.title('Types of micro expressions')
plt.show()