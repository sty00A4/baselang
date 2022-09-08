# Baselang

#### Introduction

Baselang is a high level, dynamically typed programming language written completely in python. The whole language is
compressed in one file
`baselang.py`.

#### Syntax

The grammar has some Lua-inspired syntax with the keywords
`then`, `do` and `end`, but also has statements like `a = 1` which would assign the number `1` to the variable `a`.

The whole grammar is explained in the `grammar.txt` file, though it might a bit hard to understand at first

| TYPES   | example                   |
|---------|---------------------------|
| number  | `1, 2, 6.9, ...`          |
| null    | `null`                    |
| boolean | `true / false`            |
| string  | `"text..."`               |
| list    | `[4, false, "text", ...]` |
| table   | `{"a": 1, "b": true}`     |

#### Assigment

`[identifier] = [expression]` >>>
returns `[value]` from `[expression]`.

*no full tutorial*
