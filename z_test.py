class A:
    i = 10

    def __init__(self):
        print(self.i)

a = A()
print(type(A.i))
b = A()
print(a.i)