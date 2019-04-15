class father(object):
    def getvalue(self):
        pass

    def show(self):
        print(self.getvalue())


class son(father):
    def getvalue(self):
        return 10

if __name__ == "__main__":
    s = son()
    s.show()
