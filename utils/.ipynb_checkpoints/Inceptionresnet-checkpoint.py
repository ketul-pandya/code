import datetime


def seprator():
    print("----------------------------------------------")



num1 = [1, 2, 3, 5, 7, 8, 9, 10];

even_count = len(list(filter(lambda x:(x%2==0),num1)))

num =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even = list(filter(lambda x:x%2==0,num))

odd = list(filter(lambda x:x%2!=0,num))

square = list(map(lambda x:x**2,num))

cube = list(map(lambda x:x**3,num))
a = 'Prince'
sfun = lambda x:True if x.startswith('P') else False

now = datetime.datetime.now()

year = lambda x:x.year
month = lambda y:y.month
day = lambda z:z.day



seprator()
print(even_count)
seprator()
print(year(now))
seprator()
print(month(now))
seprator()
print(day(now))

seprator()
print(cube)
seprator()
print(square)
seprator()
print(even)
seprator()
print(odd)
seprator()
print(sfun(a))
seprator()