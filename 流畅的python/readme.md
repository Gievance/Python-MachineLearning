# Python补漏

## 元组

**创建元组**
```commandline
>>> tup1 = ('Google', 'Runoob', 1997, 2000)
>>> tup2 = (1, 2, 3, 4, 5 )
>>> tup3 = "a", "b", "c", "d"   #  不需要括号也可以
>>> type(tup3)
<class 'tuple'>
```
`创建空元组`
>tup1 = ()

**当元组中只包含一个元素时，需要在元素后面添加逗号 , ，否则括号会被当作运算符使用：**
```commandline
>>> tup1 = (50)
>>> type(tup1)     # 不加逗号，类型为整型
<class 'int'>

>>> tup1 = (50,)
>>> type(tup1)     # 加上逗号，类型为元组
<class 'tuple'>
```


**访问元组**

`元组与字符串类似，下标索引从 0 开始，可以进行截取，组合等。`

`元组可以使用下标索引来访问元组中的值，如下实例:`
```commandline
tup1 = ('Google', 'Runoob', 1997, 2000)
tup2 = (1, 2, 3, 4, 5, 6, 7 )
 
print ("tup1[0]: ", tup1[0])
print ("tup2[1:5]: ", tup2[1:5])
```

**修改元组**
`元组中的元素值是不允许修改的，但我们可以对元组进行连接组合，如下实例:`
```commandline
#!/usr/bin/python3
 
tup1 = (12, 34.56)
tup2 = ('abc', 'xyz')
 
# 以下修改元组元素操作是非法的。
# tup1[0] = 100
 
# 创建一个新的元组
tup3 = tup1 + tup2
print (tup3)
```
**删除元组**

`元组中的元素值是不允许删除的，但我们可以使用del语句来删除整个元组，如下实例:`
```commandline
tup = ('Google', 'Runoob', 1997, 2000)
 
print (tup)
del tup
print ("删除后的元组 tup : ")
print (tup)

以上实例元组被删除后，输出变量会有异常信息，输出如下所示：
删除后的元组 tup : 
Traceback (most recent call last):
  File "test.py", line 8, in <module>
    print (tup)
NameError: name 'tup' is not defined

```
**元组运算符**

`与字符串一样，元组之间可以使用 +、+=和 * 号进行运算。这就意味着他们可以组合和复制，运算后会生成一个新的元组。`
a = (1,2,3)
b = (2,3,4)
- len(a) ,计算元素个数
- c=a+b ,组合一个新元组
- a+=b ,就地修改
- a*2 ,复制
- 3 in a ,判断存在
- for x in a:..  ,迭代

### 命名元组

`collections.namedtuple(typename, field_names, verbose=False, rename=False) `
返回一个具名元组子类 typename，其中参数的意义如下：

    typename：元组名称
    field_names: 元组中元素的名称
    rename: 如果元素名称中含有 python 的关键字，则必须设置为 rename=True
    verbose: 默认就好 

下面来看看声明一个具名元组及其实例化的方法：
```
import collections

# 两种方法来给 namedtuple 定义方法名
#User = collections.namedtuple('User', ['name', 'age', 'id'])
User = collections.namedtuple('User', 'name age id')
user = User('tester', '22', '464643123')

print(user)
```
`具名元组的特有属性:`

类属性 
- _fields：包含这个类所有字段名 ;
- _make(iterable)：接受一个可迭代对象来生产这个类的实例 实例方法 ;
- _asdict()：把具名元组以 collections.OrdereDict 的形式返回，可以利用它来把元组里的信息友好的展示出来 ;
```python
from collections import namedtuple

# 定义一个namedtuple类型User，并包含name，sex和age属性。
User = namedtuple('User', ['name', 'sex', 'age'])

# 创建一个User对象
user = User(name='Runoob', sex='male', age=12)

# 获取所有字段名
print( user._fields )

# 也可以通过一个list来创建一个User对象，这里注意需要使用"_make"方法
user = User._make(['Runoob', 'male', 12])

print( user )
# User(name='user1', sex='male', age=12)

# 获取用户的属性
print( user.name )
print( user.sex )
print( user.age )

# 修改对象属性，注意要使用"_replace"方法
user = user._replace(age=22)
print( user )
# User(name='user1', sex='male', age=21)

# 将User对象转换成字典，注意要使用"_asdict"
print( user._asdict() )
# OrderedDict([('name', 'Runoob'), ('sex', 'male'), ('age', 22)])
```
Output:
```commandline
('name', 'sex', 'age')
User(name='Runoob', sex='male', age=12)
Runoob
male
12
User(name='Runoob', sex='male', age=22)
OrderedDict([('name', 'Runoob'), ('sex', 'male'), ('age', 22)])
```


## 字典

字典的每个键值 key=>value 对用冒号 : 分割，每个对之间用逗号(,)分割，整个字典包括在花括号 {} 中 ,格式如下所示： 

`d = {key1 : value1, key2 : value2, key3 : value3 }`

注意：dict 作为 Python 的关键字和内置函数，变量名不建议命名为 dict。

**创建空字典**
```
# 使用大括号 {} 来创建空字典
emptyDict = {}
# 使用内置函数创建空字典
emptyDict = dict()

# 打印字典
print(emptyDict)

# 查看字典的数量
print("Length:", len(emptyDict))

# 查看类型
print(type(emptyDict))
```
**访问字典**

`把相应的键放入到方括号中，如下实例:`
```
tinydict = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}
 
print ("tinydict['Name']: ", tinydict['Name'])
print ("tinydict['Age']: ", tinydict['Age'])
```
**修改字典**

`向字典添加新内容的方法是增加新的键/值对，修改或删除已有键/值对`

```
tinydict = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}
 
tinydict['Age'] = 8               # 更新 Age
tinydict['School'] = "菜鸟教程"  # 添加信息
```

**删除字典元素**

`能删单一的元素也能清空字典，清空只需一项操作。显式删除一个字典用del命令，如下实例：`

```python
tinydict = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}
 
del tinydict['Name'] # 删除键 'Name'
tinydict.clear()     # 清空字典
del tinydict         # 删除字典

```
**字典键的特性**

`字典值可以是任何的 python 对象，既可以是标准的对象，也可以是用户定义的，但键不行。`

1. 不允许同一个键出现两次。创建时如果同一个键被赋值两次，后一个值会被记住，如下实例：
```python
tinydict = {'Name': 'Runoob', 'Age': 7, 'Name': '小菜鸟'}
 
print ("tinydict['Name']: ", tinydict['Name'])
# tinydict['Name']:  小菜鸟
```
2. 键必须不可变，所以可以用数字，字符串或元组充当，而用列表就不行，如下实例：
```python
tinydict = {['Name']: 'Runoob', 'Age': 7}
 
print ("tinydict['Name']: ", tinydict['Name'])
"""
Traceback (most recent call last):
  File "test.py", line 3, in <module>
    tinydict = {['Name']: 'Runoob', 'Age': 7}
TypeError: unhashable type: 'list'
"""
```

**内置函数**
- len(dict)
- str(dict)
- type(variable)

**内置方法**
- dict.clear()  删除字典内所有元素
- dict.copy()   返回一个字典的浅复制
- dict.fromkeys(seq[,value])  用于创建一个新字典，以序列 seq 中元素做字典的键，value 为字典所有键对应的初始值。
- dict.get(key,default=None)  返回指定键的值，如果键不在字典中返回 default 设置的默认值
- dict.items() 以列表返回视图对象，是一个可遍历的key/value 对。
- dict.keys()  返回键的视图对象
- dict.values() 返回值的视图对象
- dict.setdefault(key,default=None) 如果键不存在于字典中，将会添加键并将值设为默认值。
- dict.update(dict2) 把字典dict2的键/值对更新到dict里
- pop(key,[defalut])  删除字典 key（键）所对应的值，返回被删除的值。
  - 如果 key 存在 - 删除字典中对应的元素
  - 如果 key 不存在 - 返回设置指定的默认值 default
  - 如果 key 不存在且默认值 default 没有指定 - 触发 KeyError 异常

- popitem() 返回并删除字典中的最后一对键和值
  - 如果字典已经为空，却调用了此方法，就报出 KeyError 异常。



## 集合
` 无序的不重复元素序列。`

`集合中的元素不会重复，并且可以进行交集、并集、差集等常见的集合操作。`

`可以使用大括号 { } 创建集合，元素之间用逗号 , 分隔， 或者也可以使用 set() 函数创建集合。`

**创建集合**
```python
parame = {value01,value02,...}
或者
set(value)
```
`注意`：创建一个空集合必须用 set() 而不是 { }，因为 { } 是用来创建一个空字典。 
```python
1.创建一个含有一个元素的集合

>>> my_set = set(('apple',))
>>> my_set
{'apple'}

2.创建一个含有多个元素的集合

>>> my_set = set(('apple','pear','banana'))
>>> my_set
{'apple', 'banana', 'pear'}

3.如无必要，不要写成如下形式

>>> my_set = set('apple') # 将字符串视为迭代
>>> my_set
{'l', 'e', 'p', 'a'}
>>> my_set1 = set(('apple')) # 不加逗号，等同上句
>>> my_set1
{'l', 'e', 'p', 'a'}
```



**集合操作**
1. 添加元素
   ```python
   s.add( x ) # 将元素 x 添加到集合 s 中，如果元素已存在，则不进行任何操作
   s.update( x ) # 还有一个方法，也可以添加元素，且参数可以是列表，元组，字典等
   ```
   ```python
    s.update( {"字符串"} ) 将字符串添加到集合中，有重复的会忽略。
    s.update( "字符串" ) 将字符串拆分单个字符后，然后再一个个添加到集合中，有重复的会忽略。 

    >>> thisset = set(("Google", "Runoob", "Taobao"))
    >>> print(thisset)
    {'Google', 'Runoob', 'Taobao'}
    >>> thisset.update({"Facebook"})
    >>> print(thisset) 
    {'Google', 'Runoob', 'Taobao', 'Facebook'}
    >>> thisset.update("Yahoo")
    >>> print(thisset)
    {'h', 'o', 'Facebook', 'Google', 'Y', 'Runoob', 'Taobao', 'a'}
    >>>
    ```     
2. 移除元素
   ```python
   s.remove( x ) # 将元素 x 从集合 s 中移除，如果元素不存在，则会发生错误
   s.discard( x ) # 还有一个方法也是移除集合中的元素，且如果元素不存在，不会发生错误
   s.pop() # 设置随机删除集合中的一个元素(set 集合的 pop 方法会对集合进行无序的排列，然后将这个无序排列集合的左面第一个元素进行删除)
   ```
   计算集合元素个数
   ```python
   len(s)
   ```
   
3. 清空集合
    ```python
    s.clear()
    ```
   
4. 判断元素是否在集合中存在
    ```python
    x in s
    ```
   



**内置方法**

- add()为集合添加元素 
- clear()	移除集合中的所有元素 
- copy()	拷贝一个集合 
- difference()	返回多个集合的差集 
- difference_update()	移除集合中的元素，该元素在指定的集合也存在。 
- discard()	删除集合中指定的元素 
- intersection()	返回集合的交集 
- intersection_update()	返回集合的交集。 
- isdisjoint()	判断两个集合是否包含相同的元素，如果没有返回 True，否则返回 False。 
- issubset()	判断指定集合是否为该方法参数集合的子集。 
- issuperset()	判断该方法的参数集合是否为指定集合的子集 
- pop()	随机移除元素 
- remove()	移除指定元素 
- symmetric_difference()	返回两个集合中不重复的元素集合。 
- symmetric_difference_update()	移除当前集合中在另外一个指定集合相同的元素，并将另外一个指定集合中不同的元素插入到当前集合中。 
- union()	返回两个集合的并集 
- update()	给集合添加元素 
- len()	计算集合元素个数