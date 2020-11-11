class C:
    def __init__(self):
        self.a = 12

        self.c = 33
        self.b = self.get_2c()

    def get_2c(self):
        x = self.c
        return x*2


demo = C()
print(demo.c)