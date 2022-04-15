# Basepy

#### INTRODUCTION

Basepy is a high level, dynamically typed programming language written completely in python. The whole language is
compressed in one file
`basepy.py`.

#### SYNTAX

The grammar has some Lua-inspired syntax with the keywords
`then`, `do` and `end`, but also has statements like `var a = 1` which would assign the number `1` to the variable `a`.

The whole grammar is explained in the `grammar.txt` file, though it might a bit hard to understand at first

| TYPES   | example                   |
|---------|---------------------------|
| number  | `1, 2, 6.9, ...`          |
| null    | `null`                    |
| boolean | `true / false`            |
| string  | `"text..."`               |
| list    | `[4, false, "text", ...]` |

VARIABLE ASSIGNMENT

`var [name] = [expression]` >>>
returns `[value]` from `[expression]`.

IF STATEMENT

`if [expression1] then [expression2] ` >>>
returns _expression2_ if _expression1_'s _value_ is _true_.

`if [expression1] then [expression2] else [expression3]` >>>
returns _value2_ of _expression2_ if _expression1_'s _value1_
is _true_ otherwise _value3_ of _expression3_.

`if [expression1] then [expression2] elif [expression3] ...`
(` else [expression4]`) >>>
returns _value2_ of _expression2_ if _expression1_'s _value1_
is _true_. If it's _false_, it's going to check all the _expressions_
until one is true, otherwise _value4_ of _expression4_ will be returned
if given.

if-statements don't have to return something, they can also execute
multiple statements:
```
if [expression]

    [statements]
    
end
```
the `end` keyword is important here
```
if [expression]

    [statements]
    
elif [expression]

    [statements]

...

else [expression]

    [statements]
    
end
```
